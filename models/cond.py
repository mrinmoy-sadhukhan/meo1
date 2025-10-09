import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.misc import FrozenBatchNorm2d
from einops import rearrange
from models.custom_modules.cond_detr_decoder_layer import ConditionalDecoderLayer


class ConditionalDETR(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_classes=92,
        n_tokens=225,
        n_layers=6,
        n_heads=8,
        n_queries=100,
        use_frozen_bn=False,
    ):
        super().__init__()

        self.backbone = create_feature_extractor(
            torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True),
            return_nodes={"layer4": "layer4"},
        )

        if use_frozen_bn:
            self.replace_batchnorm(self.backbone)

        self.conv1x1 = nn.Conv2d(2048, d_model, kernel_size=1, stride=1)
        self.pe_encoder = nn.Parameter(
            torch.rand((1, n_tokens, d_model)), requires_grad=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, 4 * d_model, 0.1, batch_first=True
            ),
            num_layers=n_layers,
        )

        self.queries = nn.Parameter(
            torch.rand((1, n_queries, d_model)), requires_grad=True
        )

        self.decoder_layers = nn.ModuleList(
            [ConditionalDecoderLayer(d_model, n_heads) for _ in range(n_layers)]
        )

        self.linear_class = nn.Linear(d_model, n_classes)
        self.linear_bbox = nn.Linear(d_model, 4)

    def forward(self, x):
        tokens = self.backbone(x)["layer4"]
        print(tokens.shape)
        tokens = self.conv1x1(tokens)
        tokens = rearrange(tokens, "b c h w -> b (h w) c")
        B,N,D = tokens.shape
        memory = self.transformer_encoder(tokens + self.pe_encoder)

        #object_queries = self.queries.repeat(memory.shape[0], 1, 1) ##original
        # ===== 2️⃣ Query generation =====
        # Content queries (learned embeddings)
        content_q = self.content_queries.unsqueeze(0).expand(B, -1, -1)   # [B, Qc, D]

        # Spatial queries: select top-k informative regions
        scores = self.spatial_score(memory).squeeze(-1)                   # [B, N]
        topk = min(self.topk_spatial, scores.shape[1])
        topk_vals, topk_idx = torch.topk(scores, k=topk, dim=1)
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, topk)
        spatial_feats = memory[batch_idx, topk_idx]                       # [B, topk, D]
        spatial_q = spatial_feats                                         # [B, topk, D]

        # ===== 3️⃣ Combine content + spatial queries =====
        queries = torch.cat([content_q, spatial_q], dim=1)                # [B, Q_total, D]

        
        # Object queries are the same for the first decoder layer as decoder embeddings
        decoder_embeddings = queries
        object_queries = queries
        class_preds, bbox_preds = [], []
        for layer in self.decoder_layers:
            decoder_embeddings, ref_points = layer(
                decoder_embeddings, object_queries, memory
            )
            class_preds.append(self.linear_class(decoder_embeddings))
            bbox_preds.append(self.linear_bbox(decoder_embeddings) + ref_points)

        return torch.stack(class_preds, dim=1), torch.stack(bbox_preds, dim=1)

    @staticmethod
    def replace_batchnorm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                frozen_bn = FrozenBatchNorm2d(child.num_features)
                frozen_bn.weight.data = child.weight.data
                frozen_bn.bias.data = child.bias.data
                frozen_bn.running_mean.data = child.running_mean.data
                frozen_bn.running_var.data = child.running_var.data
                setattr(module, name, frozen_bn)
            else:
                ConditionalDETR.replace_batchnorm(child)
