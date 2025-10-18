import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class DeformableSelfAttention(nn.Module):
    def __init__(self, dim, n_heads=8, n_points=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = dim // n_heads

        # learnable projection layers
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1)
        
        # predict offsets and attention weights
        self.offset_proj = nn.Conv2d(dim, 2 * n_heads * n_points, 3, padding=1)
        self.attn_weight_proj = nn.Conv2d(dim, n_heads * n_points, 3, padding=1)

        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x,H,W):
        
        x = rearrange(x, 'b q (h w) -> b q h w', h=H, w=W)
        B, C, H, W = x.shape
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=1)

        offsets = self.offset_proj(x)
        offsets = offsets.view(B, self.n_heads, self.n_points, 2, H, W)
        
        attn_weights = self.attn_weight_proj(x)
        attn_weights = attn_weights.view(B, self.n_heads, self.n_points, H, W)
        attn_weights = torch.softmax(attn_weights, dim=2)

        # normalized grid for sampling
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

        outputs = []
        for h in range(self.n_heads):
            v_h = v[:, h * self.head_dim:(h + 1) * self.head_dim]
            q_h = q[:, h * self.head_dim:(h + 1) * self.head_dim]

            sampled = 0
            for p in range(self.n_points):
                # deform the grid
                offset = offsets[:, h, p].permute(0, 2, 3, 1)  # (B, H, W, 2)
                grid = base_grid[None] + offset / torch.tensor([W / 2, H / 2], device=x.device)
                sampled_v = F.grid_sample(v_h, grid, align_corners=True)
                sampled += attn_weights[:, h, p].unsqueeze(1) * sampled_v

            outputs.append(sampled)

        out = torch.cat(outputs, dim=1)
        out = self.out_proj(out)
        out = rearrange(x, 'b q h w -> b q (h w)')
        return out
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import einsum
# from einops import rearrange

# # Helper functions
# def create_grid_like(tensor):
#     """ Creates a grid of normalized points [-1, 1] for the given tensor's height and width """
#     h, w = tensor.shape[-2], tensor.shape[-1]
#     device = tensor.device
    
#     # Create a meshgrid for h x w
#     grid_x, grid_y = torch.meshgrid(
#         torch.linspace(-1, 1, w, device=device),
#         torch.linspace(-1, 1, h, device=device),
#     )
#     grid = torch.stack((grid_x, grid_y), dim=-1)  # Shape: (h, w, 2)

#     return grid

# def normalize_grid(grid):
#     """ Normalizes grid coordinates to range from [-1, 1] """
#     b,h,w,c= grid.shape
#     grid_h, grid_w = grid[..., 0], grid[..., 1]

#     # Normalize the grid to [-1, 1]
#     grid_h = 2.0 * grid_h / (h - 1) - 1.0
#     grid_w = 2.0 * grid_w / (w - 1) - 1.0

#     return torch.stack([grid_h, grid_w], dim=-1)

# # Main Deformable Attention Class
# class DeformableSelfAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, downsample_factor=4, offset_scale=1.0, offset_groups=None):
#         super().__init__()
#         inner_dim = heads * dim_head
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.offset_groups = offset_groups if offset_groups else heads
#         self.downsample_factor = downsample_factor

#         assert (heads % self.offset_groups) == 0, "Heads must be divisible by offset groups"

#         # Convolution layers for generating queries, keys, and values
#         self.to_q = nn.Conv2d(dim, inner_dim, 1)
#         self.to_k = nn.Conv2d(dim, inner_dim, 1)
#         self.to_v = nn.Conv2d(dim, inner_dim, 1)

#         # Offset network to learn the spatial shifts (offsets) for reference points
#         offset_dim = inner_dim // self.offset_groups
#         self.offset_network = nn.Sequential(
#             nn.Conv2d(offset_dim, offset_dim, kernel_size=6, padding=2, stride=downsample_factor, groups=offset_dim),
#             nn.GELU(),
#             nn.Conv2d(offset_dim, 2, 1),  # Produces 2D offsets (dx, dy)
#             nn.Tanh(),  # Limits offsets between [-1, 1]
#         )
#         self.offset_scale = nn.Parameter(torch.tensor(offset_scale))

#         # Final output projection layer
#         self.to_out = nn.Conv2d(inner_dim, dim, 1)

#     def forward(self, x,H,W):
#         #B, C, H, W = x.shape
#         x = rearrange(x, 'b q (h w) -> b q h w', h=H, w=W) 
#         # 1. Compute Queries, Keys, Values
#         q = self.to_q(x)  # Shape: (B, heads * dim_head, H, W)
#         k = self.to_k(x)  # Shape: (B, heads * dim_head, H, W)
#         v = self.to_v(x)  # Shape: (B, heads * dim_head, H, W)

#         # 2. Generate offsets using offset network
#         grouped_q = rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.offset_groups)  # Shape: (B * offset_groups, C / offset_groups, H, W)
#         offsets = self.offset_network(grouped_q)  # Shape: (B * offset_groups, 2, h', w')

#         # 3. Create normalized grid and add learned offsets
#         grid = create_grid_like(offsets)  # Shape: (H', W', 2)
#         offset_grid = grid + offsets.permute(0, 2, 3, 1)  # Add offsets to reference grid
#         print(offset_grid.shape)
#         # 4. Normalize grid for sampling
#         normalized_grid = normalize_grid(offset_grid)

#         # 5. Sample keys and values using bilinear interpolation
#         kv_feats = F.grid_sample(
#             rearrange(x, 'b c h w -> (b g) c h w', g=self.offset_groups),  # Input feature map for sampling
#             normalized_grid.unsqueeze(1),  # Shape: (B * g, 1, H', W', 2)
#             mode='bilinear', align_corners=False
#         )  # Resulting sampled features: Shape: (B * g, c, h', w')

#         # Restore batch and groups
#         kv_feats = rearrange(kv_feats, '(b g) c h w -> b (g c) h w', b=B)

#         # 6. Compute attention scores between queries and sampled keys
#         q = rearrange(q, 'b (h d) h1 w1 -> b h (h1 w1) d', h=self.heads)
#         k = rearrange(kv_feats, 'b (h d) h1 w1 -> b h (h1 w1) d', h=self.heads)
#         v = rearrange(kv_feats, 'b (h d) h1 w1 -> b h (h1 w1) d', h=self.heads)

#         attn = einsum('b h n d, b h m d -> b h n m', q, k) * self.scale
#         attn = attn.softmax(dim=-1)

#         # 7. Compute attention-weighted output
#         out = einsum('b h n m, b h m d -> b h n d', attn, v)
#         out = rearrange(out, 'b h (h1 w1) d -> b (h d) h1 w1', h1=H, w1=W)
#         out=self.to_out(out)
#         out = rearrange(x, 'b q h w -> b q (h w)') 
#         # 8. Final output projection
#         return out