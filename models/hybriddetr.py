import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from torchvision.models.feature_extraction import create_feature_extractor
class PositionEmbeddingSine(nn.Module):
    """
    Standard 2D sine-cosine positional encoding from DETR.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x):
        """
        x: [B, C, H, W] feature map
        return: [B, H*W, 2*num_pos_feats]
        """
        B, C, H, W = x.shape
        mask = torch.zeros(B, H, W, device=x.device, dtype=torch.bool)

        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # [B, H, W, 2*num_pos_feats]

        return pos.flatten(1, 2)  # [B, H*W, D]
        
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute learnable 2D positional embeddings.
    """
    def __init__(self, num_pos_feats=256, max_h=64, max_w=64):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, num_pos_feats)
        self.col_embed = nn.Embedding(max_w, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, H*W, D]
        """
        B, C, H, W = x.shape
        i = torch.arange(W, device=x.device)
        j = torch.arange(H, device=x.device)
        x_emb = self.col_embed(i)  # [W, D]
        y_emb = self.row_embed(j)  # [H, D]
        pos = torch.cat([
            y_emb.unsqueeze(1).expand(H, W, -1),
            x_emb.unsqueeze(0).expand(H, W, -1)
        ], dim=-1)  # [H, W, 2*D]
        pos = pos.reshape(H * W, -1)
        pos = pos.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2*D]
        return pos

class SlotAttention(nn.Module):
    """
    Slot Attention module: aggregates N input features into K slot embeddings.
    Reference: Locatello et al., 2020
    """

    def __init__(self, num_slots=6, dim=256, iters=3):
        super().__init__()
        self.num_slots = num_slots      # number of slots
        self.iters = iters              # number of attention iterations
        self.scale = dim ** -0.5        # scaling factor for attention

        # Slot initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, num_slots, dim))

        # Linear maps for attention
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Slot update GRU
        self.gru = nn.GRUCell(dim, dim)

        # Feed-forward module after GRU
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim)
        )

        # Layer norms
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, N, D] input features (flattened spatial map)
        returns: [B, num_slots, D] aggregated slot embeddings
        """
        B, N, D = x.size()

        # Initialize slots
        mu = self.slots_mu.expand(B, -1, -1)                 # [B, num_slots, D]
        sigma = F.softplus(self.slots_sigma).expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(sigma)          # add noise

        # Normalize input features
        x = self.norm_input(x)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            # Compute attention
            q = self.to_q(slots_norm)     # [B, num_slots, D]
            k = self.to_k(x)              # [B, N, D]
            v = self.to_v(x)              # [B, N, D]

            # Attention logits and weights
            attn_logits = torch.matmul(k, q.transpose(-1, -2)) * self.scale  # [B, N, num_slots]
            attn = attn_logits.softmax(dim=-1)                                # softmax over slots

            # Weighted aggregation
            updates = torch.matmul(attn.transpose(-1, -2), v)  # [B, num_slots, D]

            # Slot update with GRU
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, -1, D)

            # Residual MLP
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots  # [B, num_slots, D]

class ReIDWithSlotAttention(nn.Module):
    def __init__(self, num_classes, input_dim=256, slot_dim=256, num_slots=1, emb_dim=512):
        super().__init__()
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=slot_dim)
        self.proj = nn.Linear(input_dim, slot_dim)
        self.fc = nn.Linear(num_slots * slot_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, reid_input):
        """
        reid_input: [B, topk, D]
        returns:
          reid_emb   [B, topk, emb_dim]
          reid_logits [B, topk, num_classes]
        """
        B, T, D = reid_input.shape
        reid_input = self.proj(reid_input)  # [B, topk, slot_dim]

        # Treat each [B, topk] object as a separate batch item
        x = reid_input.reshape(B * T, 1, -1)  # [B*T, N=1, slot_dim]
        slots = self.slot_attention(x)        # [B*T, num_slots, slot_dim]
        #print("slots")
        #print(slots.shape)
        emb = self.fc(slots.view(B * T, -1))  # [B*T, emb_dim]
        emb = self.norm(emb)
        emb = F.normalize(emb, p=2, dim=-1)

        logits = self.classifier(emb)         # [B*T, num_classes]

        # reshape back
        emb = emb.view(B, T, -1)              # [B, topk, emb_dim]
        logits = logits.view(B, T, -1)        # [B, topk, num_classes]

        return emb, logits


        
def inverse_sigmoid(x, eps=1e-6):
    """
    x: Tensor with values in [0,1]
    returns: logit(x) = log(x/(1-x))
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DownBlock(nn.Module):
    """Concat previous downsampled feature + current feature, then downsample"""
    def __init__(self, prev_ch, curr_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(prev_ch + curr_ch, out_ch, stride=1),  # downsample + process
            ConvBNReLU(out_ch, out_ch)
        )
    
    def forward(self, prev_down, curr):
        prev_down_ds = F.interpolate(prev_down, size=curr.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([prev_down_ds, curr], dim=1)
        x = self.block(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
    
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
# -------------------------
# Transformer Encoder (single layer)
# -------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, N, D]
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        ff = self.ffn(x)
        x = x + ff
        x = self.norm2(x)
        return x
# -------------------------
# Decoder Layer (self-attn + cross-attn placeholder for deformable)
# -------------------------
# class DecoderLayer(nn.Module):
#     def __init__(self, d_model=256, nhead=8):
#         super().__init__()
#         self.d_model = d_model
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True) ##deformable block
#         # cross-attention placeholder (use deformable attention here in production)
#         self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.ReLU(inplace=True),
#             nn.Linear(d_model * 4, d_model)
#         )
#         self.norm3 = nn.LayerNorm(d_model)
#         self.class_head = nn.Linear(d_model, 1+1)  # one person class (logit)
#         self.box_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))
#         self.dropout = nn.Dropout(0.1)

#         # --- Spatial conditioning ---
#         #self.ref_point_proj = nn.Linear(4, d_model)   # encode (cx, cy, w, h)
#         self.displacement_ffn = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(inplace=True),
#             nn.Linear(d_model, d_model)
#         )
#         self.lambda_q = nn.Parameter(torch.ones(d_model))  # learnable scaling per channel
    
#     def positional_encoding_ref(self, ref_points):
#         """
#         ref_points: [B, Q, 2] with values in [0,1] (cx, cy)
#         returns: [B, Q, d_model] positional encoding
#         """
#         # use the same pattern as DETR: half dims per axis
#         batch, qn, _ = ref_points.shape
#         device = ref_points.device
#         half_dim = self.d_model // 2
#         # create dim_t
#         dim_t = torch.arange(half_dim, dtype=torch.float32, device=device)
#         dim_t = 10000 ** (2 * (dim_t // 2) / float(half_dim))

#         # normalize just in case
#         ref = torch.sigmoid(ref_points)  # [B, Q, 2]
#         px = ref[..., 0].unsqueeze(-1) / dim_t  # [B, Q, half_dim]
#         py = ref[..., 1].unsqueeze(-1) / dim_t  # [B, Q, half_dim]

#         # interleave sin/cos
#         sx = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=-1).flatten(-2)
#         sy = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=-1).flatten(-2)

#         pos = torch.cat([sx, sy], dim=-1)  # [B, Q, d_model] (if d_model even)
#         # if d_model is odd, pad
#         if pos.shape[-1] != self.d_model:
#             pad = self.d_model - pos.shape[-1]
#             pos = F.pad(pos, (0, pad), value=0.0)
#         return pos

#     def forward(self, objectquery, fixedembeddings, ref_boxes):
#         """
#         query: [B, num_queries, D]
#         memory: [B, N, D]  (encoder features)
#         ref_boxes: [B, num_queries, 4]  (cx, cy, w, h in [0,1])
#         """
#         # query=objectquery
#         # memory=fixedembeddings
#         # # ===== 1. Self-Attention =====
#         # q1 = self.norm1(query + self.dropout(self.self_attn(query, query, query)[0]))

#         # # 2) positional encoding for reference points -> use only cx,cy
#         # ref_points = ref_boxes[..., :2]  # [B, Q, 2]
#         # pos_embed = self.positional_encoding_ref(ref_points)  # [B, Q, D]
        
#         # # ===== 2. Positional conditioning via reference boxes =====
#         # # Convert reference boxes to spatial embeddings
#         # #ref_embed = torch.sigmoid(self.ref_point_proj(ref_boxes))  # [B, Q, D]
#         # scale=0.1
#         # displacement = torch.tanh(self.displacement_ffn(q1))+scale                   # learned shift
#         # spatial_query = displacement * (self.lambda_q.view(1, 1, -1) * pos_embed) # element-wise modulation

#         # # ===== 3. Cross-Attention =====
#         # cross_query = q1 + spatial_query
#         # cross_out = self.cross_attn(cross_query, memory, memory)[0]
#         # q2 = self.norm2(q1 + self.dropout(cross_out))

#         # # ===== 4. Feedforward =====
#         # q3 = self.norm3(q2 + self.dropout(self.ffn(q2)))

#         # # ===== 5. Output predictions =====
#         # class_logits = self.class_head(q3)
#         # box_deltas = torch.tanh(self.box_head(q3))*0.1

#         # return q3, class_logits, box_deltas
#         query = objectquery
#         memory = fixedembeddings

#         # ===== 1️⃣ Self-Attention =====
#         # Queries interact with each other (object relations)
#         self_attn_out = self.self_attn(query, query, query)[0]
#         q1 = self.norm1(query + self.dropout(self_attn_out))

#         # ===== 2️⃣ Positional Encoding from Reference Boxes =====
#         # Use only (cx, cy) for position embedding
#         ref_points = ref_boxes[..., :2]  # [B, Q, 2]
#         pos_embed = self.positional_encoding_ref(ref_points)  # [B, Q, D]

#         # ===== 3️⃣ Learnable Spatial Modulation =====
#         # Apply displacement based on query content
#         # Use multiplicative form to keep updates stable (~1× mean)
#         scale = 0.1
#         displacement = 1 + scale * torch.tanh(self.displacement_ffn(q1))  # [B, Q, D]
#         spatial_query = displacement * (self.lambda_q.view(1, 1, -1) * pos_embed)

#         # ===== 4️⃣ Cross-Attention with Encoder Features =====
#         # Inject geometric prior into query before cross-attention
#         cross_query = q1 + spatial_query
#         cross_out = self.cross_attn(cross_query, memory, memory)[0]
#         q2 = self.norm2(q1 + self.dropout(cross_out))

#         # ===== 5️⃣ Feed-Forward Network =====
#         ffn_out = self.ffn(q2)
#         q3 = self.norm3(q2 + self.dropout(ffn_out))

#         # ===== 6️⃣ Prediction Heads =====
#         # Classification head (logits)
#         class_logits = self.class_head(q3)  # [B, Q, num_classes]

#         # Bounding box delta prediction
#         box_deltas = torch.tanh(self.box_head(q3)) * 0.1  # small offset

#         # ===== 7️⃣ Bounding Box Refinement =====
#         # Use inverse-sigmoid trick for numerically stable refinement
#         updated_boxes = torch.sigmoid(
#             inverse_sigmoid(ref_boxes) + box_deltas
#         )

#         return q3, class_logits, updated_boxes
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # 1️⃣ Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 2️⃣ Cross-Attention (queries → encoder memory)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 3️⃣ Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        # 4️⃣ Prediction heads
        self.class_head = nn.Linear(d_model, 91)    # 91 = COCO classes (modify)
        self.bbox_head = nn.Linear(d_model, 4)

        # Reference box modulator
        self.ref_proj = nn.Linear(4, d_model)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, query, memory, ref_boxes):
        """
        query:      [B, Q, D]
        memory:     [B, N, D]
        ref_boxes:  [B, Q, 4]   (cx, cy, w, h normalized)
        """
        B, Q, D = query.shape

        # ===== 1️⃣ Self-attention (content refinement) =====
        q = query.transpose(0, 1)  # [Q, B, D]
        q2 = self.self_attn(q, q, q)[0].transpose(0, 1)
        query = self.norm1(query + self.dropout1(q2))

        # ===== 2️⃣ Positional conditioning from ref_boxes =====
        pos_embed = self.ref_proj(ref_boxes) * self.scale
        q_cross = query + pos_embed

        # ===== 3️⃣ Cross-attention (query attends to memory) =====
        q_ = q_cross.transpose(0, 1)
        k_ = memory.transpose(0, 1)
        cross_out = self.cross_attn(q_, k_, k_)[0].transpose(0, 1)
        query = self.norm2(query + self.dropout2(cross_out))

        # ===== 4️⃣ Feedforward network =====
        ff = self.linear2(F.relu(self.linear1(query)))
        query = self.norm3(query + self.dropout3(ff))

        # ===== 5️⃣ Prediction =====
        cls_logits = self.class_head(query)
        box_deltas = torch.tanh(self.bbox_head(query)) * 0.05  # bounded update

        # ===== 6️⃣ Update reference boxes =====
        updated_boxes = ref_boxes.clone()
        updated_boxes[..., :2] = (ref_boxes[..., :2] + box_deltas[..., :2]).clamp(0, 1)
        updated_boxes[..., 2:] = (ref_boxes[..., 2:] * torch.exp(box_deltas[..., 2:])).clamp(1e-3, 1.0)

        return query, cls_logits, updated_boxes

class SwinUNetMultiUp(nn.Module):
    def __init__(self, swin_model_name="swin_large_patch4_window12_384",num_queries=200, topk_spatial=100,pos_type="normal", num_decoder_layers=6, pretrained=True, d_model=256):
        super().__init__()
        #self.backbone = timm.create_model(swin_model_name,pretrained=pretrained,features_only=True)
        self.backbone = timm.create_model(
               "vit_base_patch14_dinov2.lvd142m",
                pretrained=True,
                features_only=True,
                dynamic_img_size=True,
                out_indices=(1, 2, 3)
                )
        # --- Positional Encoding ---
        if pos_type == "sine":
            self.position_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)
        elif pos_type == "learned":
            self.position_encoding = PositionEmbeddingLearned(num_pos_feats=d_model // 2)
        else:
            self.position_encoding = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        channels = self.backbone.feature_info.channels()
        print(f"[INFO] channels: {channels}")
        
        self.pool2 = nn.Identity()  # same dimension
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)  # downsample ×2
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)  # downsample ×4
        
        # Project Swin stages to d_model
        self.conv2 = nn.Conv2d(channels[0], d_model, kernel_size=1)
        self.conv3 = nn.Conv2d(channels[1], d_model, kernel_size=1)
        self.conv4 = nn.Conv2d(channels[2], d_model, kernel_size=1)

        # Down path
        self.down3 = DownBlock(d_model, d_model, d_model)  # p3 + d2
        self.down4 = DownBlock(d_model, d_model, d_model)  # p4 + d3

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNReLU(d_model, d_model),
            ConvBNReLU(d_model, d_model)
        )

        # Up path
        self.up4 = UpBlock(d_model, d_model, d_model)
        self.up3 = UpBlock(d_model, d_model, d_model)
        self.up2 = UpBlock(d_model, d_model, d_model)

        # Final fusion of upsampled features
        self.fusion_proj = ConvBNReLU(3 * d_model, d_model)
        #self.encoder = TransformerEncoder(d_model=d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024),
        num_layers=6,   # default
        )
        # Query generation (content + spatial top-k)
        self.num_content_queries = num_queries
        self.content_queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.spatial_score = nn.Linear(d_model, 1)
        self.topk_spatial = topk_spatial
        self.box_init = nn.Linear(d_model, 4)
        #self._init_box_init()

        # Decoder stack
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model=d_model, nhead=8) for _ in range(num_decoder_layers)])

        # box refinement projection (optional)
        self.refine_proj = nn.Linear(d_model, 4) 
        self.reid_head = ReIDWithSlotAttention(num_classes=1000, slot_dim=256, num_slots=6)
    def _init_box_init(self):
        """Initialize box_init to produce small initial boxes."""
        nn.init.constant_(self.box_init.weight, 0.0)
        nn.init.constant_(self.box_init.bias, 0.0)
        with torch.no_grad():
            # sigmoid(-2.5) ≈ 0.08 → small width/height
            self.box_init.bias[2:].fill_(-2.5)
    def forward(self, x):
        # Swin features
        features = self.backbone(x)
        #print(features[0].shape,features[1].shape,features[2].shape)
        for i in range(0, 3): #1 to 4
            if features[i].shape[1] < features[i].shape[-1]:  # channels last
                features[i] = features[i].permute(0, 3, 1, 2).contiguous()
        
        # Feature projection using Unet
        p2 = self.conv2(self.pool2(features[0]))
        p3 = self.conv3(self.pool3(features[1]))
        p4 = self.conv4(self.pool4(features[2]))
        # Down path
        d2 = p2
        d3 = self.down3(d2, p3)
        d4 = self.down4(d3, p4)
        # Bottleneck
        bn = self.bottleneck(d4)
        # Up path
        up4 = self.up4(bn, d4)
        up3 = self.up3(up4, d3)
        up2 = self.up2(up3, d2)
        # Multi-scale upsample fusion
        up4_up = F.interpolate(up4, size=up2.shape[-2:], mode='bilinear', align_corners=False)
        up3_up = F.interpolate(up3, size=up2.shape[-2:], mode='bilinear', align_corners=False)
        cat = torch.cat([up4_up, up3_up, up2], dim=1)  # [B, 3*d_model, H, W]
        # Final projection
        out = self.fusion_proj(cat)
        
        # # Flatten for transformer input
        # B, D, H, W = out.shape
        # memory = out.flatten(2).transpose(1, 2)  # [B, H*W, D==B,N,D]
        # # Positional embedding
        # pos_embed = self.position_encoding(memory)
        # memory = memory + pos_embed
        # memory=self.encoder(memory)

        # # Query generation: content + spatial
        # # content queries
        # content_q = self.content_queries.unsqueeze(0).expand(B, -1, -1)  # [B, Qc, D]
        # # compute spatial scores on memory
        # scores = self.spatial_score(memory).squeeze(-1)  # [B, N]
        # topk = min(self.topk_spatial, scores.shape[1])
        # topk_vals, topk_idx = torch.topk(scores, k=topk, dim=1)
        # batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, topk)
        # spatial_feats = memory[batch_idx, topk_idx]  # [B, topk, D]
        # spatial_q = spatial_feats  # optionally project
        # # Combine queries
        # queries = torch.cat([content_q, spatial_q], dim=1)  # [B, Q_total, D]
        
        # # initialize reference boxes: content small centered, spatial as init_boxes
        # #init_boxes = torch.sigmoid(self.box_init(spatial_feats))  # [B, topk, 4]
        # #content_boxes = torch.tensor([0.5, 0.5, 0.05, 0.05], device=x.device).view(1,1,4).expand(B, content_q.shape[1], 4)
        # #ref_boxes = torch.cat([content_boxes, init_boxes], dim=1)  # [B, Q_total, 4]

        # # Compute (cx, cy) from topk_idx grid positions
        # # assume memory is flatten(H, W)
        # cy = (topk_idx // W).float() / H
        # cx = (topk_idx % W).float() / W
        # centers = torch.stack([cx, cy], dim=-1)  # [B, topk, 2]

        # # # Initialize small width/height
        # # wh = torch.full_like(centers, 0.08, device=x.device)
        # # spatial_boxes = torch.cat([centers, wh], dim=-1)  # [B, topk, 4]

        # # # === Optional small refinement from box_init MLP ===
        # # init_delta = torch.sigmoid(self.box_init(spatial_feats)) - 0.5  # centered offset
        # # spatial_boxes = (spatial_boxes + 0.1 * init_delta).clamp(0, 1)  # small local adjustment

        # # === Combine content + spatial queries ===
        # #queries = torch.cat([content_q, spatial_feats], dim=1)  # [B, Q_total, D]

        # # 2. Initialize boxes around centers
        # init_boxes = torch.sigmoid(self.box_init(spatial_feats))
        # # Add centers to first 2 dims only
        # new_centers = 0.08 * (init_boxes[..., :2] - 0.5) + centers

        # # Scale w,h separately (keep small)
        # new_wh = 0.05 + 0.07 * init_boxes[..., 2:]  # maps sigmoid output to [0.05, 0.12]
        # # Concatenate to create new init_boxes (out-of-place)
        # init_boxes = torch.cat([new_centers, new_wh], dim=-1)
        # # 3. Optional tiny jitter
        # init_boxes = init_boxes + (torch.rand_like(init_boxes) - 0.5) * 0.01
        # # === Reference boxes ===
        # content_boxes = torch.tensor(
        #     [0.5, 0.5, 0.05, 0.05],
        #     device=x.device
        # ).view(1, 1, 4).expand(B, content_q.shape[1], 4)

        # ref_boxes = torch.cat([content_boxes, init_boxes], dim=1)  # [B, Q_total, 4]
        # Decode with iterative refinement
        # ===== 1️⃣ Flatten encoder features =====
        B, D, H, W = out.shape
        memory = out.flatten(2).transpose(1, 2)        # [B, H*W, D]
        pos_embed = self.position_encoding(memory)
        memory = memory + pos_embed
        memory = self.encoder(memory)

        # ===== 2️⃣ Query generation =====
        # Content queries (learned embeddings)
        content_q = self.content_queries.unsqueeze(0).expand(B, -1, -1)   # [B, Qc, D]

        # Spatial queries: select top-k informative regions
        scores = self.spatial_score(memory).squeeze(-1)                   # [B, N]
        topk = min(self.topk_spatial, scores.shape[1])
        topk_vals, topk_idx = torch.topk(scores, k=topk, dim=1)
        batch_idx = torch.arange(B, device=out.device).unsqueeze(1).expand(-1, topk)
        spatial_feats = memory[batch_idx, topk_idx]                       # [B, topk, D]
        spatial_q = spatial_feats                                         # [B, topk, D]

        # ===== 3️⃣ Combine content + spatial queries =====
        queries = torch.cat([content_q, spatial_q], dim=1)                # [B, Q_total, D]

        # ===== 4️⃣ Reference box initialization =====
        # Compute (cx, cy) from grid indices
        cy = (topk_idx // W).float() / H
        cx = (topk_idx %  W).float() / W
        centers = torch.stack([cx, cy], dim=-1)                           # [B, topk, 2]

        # Initial width/height mapping — keep small and positive
        # Range roughly [0.02, 0.05] for stability
        init_raw = torch.sigmoid(self.box_init(spatial_feats))            # [B, topk, 4]
        new_centers = centers + 0.02 * (init_raw[..., :2] - 0.5)          # small local shift
        new_centers = new_centers.clamp(0.01, 0.99)                       # keep valid range
        new_wh = 0.02 + 0.03 * init_raw[..., 2:]                          # [0.02, 0.05]
        init_boxes = torch.cat([new_centers, new_wh], dim=-1)             # [B, topk, 4]

        # Small Gaussian jitter (only after first 2 epochs, if you track epoch)
        if hasattr(self, "current_epoch") and self.current_epoch > 2:
            noise = 0.005 * torch.randn_like(init_boxes)
            init_boxes = (init_boxes + noise).clamp(0.0, 1.0)

        # ===== 5️⃣ Content query reference boxes (centered small) =====
        content_boxes = torch.tensor(
            [0.5, 0.5, 0.03, 0.03], device=out.device
        ).view(1, 1, 4).expand(B, content_q.shape[1], 4)

        # ===== 6️⃣ Final reference boxes =====
        ref_boxes = torch.cat([content_boxes, init_boxes], dim=1).clamp(0.0, 1.0)  # [B, Q_total, 4]
        q = queries
        boxes = ref_boxes
        #print(q.shape)
        #print(boxes.shape)
        all_logits = []
        all_boxes = []
        for layer in self.decoder_layers:
            q, logits, boxes = layer(q, memory, boxes)
            #boxes = (inverse_sigmoid(boxes) + deltas).sigmoid()
            all_logits.append(logits)
            all_boxes.append(boxes)
        
        # # ======== Vectorized TOP-K SELECTION FOR ReID ========
        # final_logits = all_logits[-1]      # [B, Q, num_classes_det]
        # final_boxes  = all_boxes[-1]       # [B, Q, 4]
        
        # scores = final_logits.softmax(-1)[..., 1]   # [B, Q], use class-1 prob or max prob
        # topk_vals, topk_idx = torch.topk(scores, k=self.topk_spatial, dim=1)
        
        # # Gather top-k embeddings
        # reid_input = q.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, q.size(-1)))
        
        # # Feed top-k queries into ReID head
        # reid_emb, reid_logits = self.reid_head(reid_input)  # [B, topk, embedding_dim], [B, topk, num_classes]
        
        #for i, b in enumerate(all_logits):
        #    print(f"Layer {i} boxes shape: {b.shape}")
        #for i, b in enumerate(all_boxes):
        #    print(f"Layer {i} boxes shape: {b.shape}")
        #print(reid_input.shape)
        #print(reid_emb.shape)
        #print(reid_logits.shape)
        # return {
        #     'per_layer_logits': all_logits,
        #     'per_layer_boxes': all_boxes,
        #     'final_logits': all_logits[-1],
        #     'final_boxes': all_boxes[-1],
        #     'init_boxes': ref_boxes,
        #     'topk_idx': topk_idx,
        # #    'reid_emb': reid_emd,
        # #    'reid_logits': reid_logits,
        # }
        # Stack and return
        class_preds = torch.stack(all_logits, dim=1)
        bbox_preds = torch.stack(all_boxes, dim=1)
        return class_preds, bbox_preds
