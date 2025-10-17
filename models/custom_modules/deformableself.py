import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------
# Deformable Self-Attention
# -----------------------
class DeformableSelfAttention(nn.Module):
    """
    Self-attention where each query attends only to a small local neighborhood around itself.
    This is implemented as a deformable-style attention over a sequence reshaped as 2D.
    """
    def __init__(self, d_model, n_heads=8, n_points=7, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_points = n_points
        self.dropout = nn.Dropout(dropout)

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Predict local offsets for deformable attention
        self.offset_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * n_points),  # relative offsets (dx, dy) for n_points
            nn.Tanh()  # normalized offsets
        )

    def forward(self, x, H, W):
        """
        x: (B, N, d_model)
        H, W: spatial dimensions (so N = H*W)
        """
        B, N, D = x.shape
        device = x.device
        n_points = self.n_points

        # Linear projections
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute offsets for each query
        offsets = self.offset_predictor(x).view(B, N, n_points, 2)  # (B, N, n_points, 2)

        # Convert sequence indices to 2D coordinates
        coords_y = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).flatten()  # (N,)
        coords_x = torch.arange(W, device=device).repeat(H)  # (N,)
        coords = torch.stack([coords_x, coords_y], dim=-1).unsqueeze(0).unsqueeze(0).float()  # (1,1,N,2)
        coords = coords.repeat(B, self.n_heads, 1, 1)  # (B, heads, N,2)

        # Apply offsets (scaled)
        offsets_scaled = offsets.unsqueeze(1) * (2.0 / max(H, W))  # normalize to [-1,1]
        sampling_coords = coords.unsqueeze(3) + offsets_scaled.unsqueeze(2)  # (B, heads, N, n_points, 2)

        # Flatten sampling coords
        sampling_coords_flat = sampling_coords.view(B*self.n_heads, N*n_points, 2)

        # Flatten keys and values per head
        k_flat = k.transpose(1,2).contiguous().view(B*self.n_heads, N, self.head_dim)
        v_flat = v.transpose(1,2).contiguous().view(B*self.n_heads, N, self.head_dim)

        # Compute attention scores (q to k at sampled positions)
        # First, expand q to match n_points
        q_expanded = q.unsqueeze(3).repeat(1,1,1,n_points,1).view(B*self.n_heads, N*n_points, self.head_dim)

        # Compute attention: cosine similarity could be used, but simple dot-product with nearest neighbor for now
        # For simplicity, let's use dot-product with K at sampled positions using nearest neighbor approximation
        # Round sampling_coords_flat to nearest indices
        idx_x = sampling_coords_flat[...,0].round().clamp(0,W-1).long()
        idx_y = sampling_coords_flat[...,1].round().clamp(0,H-1).long()
        idx_flat = idx_y * W + idx_x  # (B*heads, N*n_points)
        k_sampled = k_flat.gather(1, idx_flat.unsqueeze(-1).expand(-1,-1,self.head_dim))  # (B*heads, N*n_points, head_dim)
        v_sampled = v_flat.gather(1, idx_flat.unsqueeze(-1).expand(-1,-1,self.head_dim))

        attn_scores = (q_expanded * k_sampled).sum(-1) / (self.head_dim ** 0.5)  # (B*heads, N*n_points)
        attn_scores = attn_scores.view(B, self.n_heads, N, n_points)
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out_heads = (attn.unsqueeze(-1) * v_sampled.view(B, self.n_heads, N, n_points, self.head_dim)).sum(3)
        out = out_heads.transpose(1,2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out
