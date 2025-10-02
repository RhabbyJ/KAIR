"""
ELSA (Enhanced Local Self-Attention) implementation for SwinIR
Adapted from the official ELSA repository: https://github.com/damo-cv/ELSA

This module provides the core ELSA attention mechanism that enhances local self-attention
with Hadamard attention and ghost head for better spatial feature modeling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


def elsa_op(
    features,
    ghost_mul,
    ghost_add,
    h_attn,
    lam,
    gamma,
    kernel_size=5,
    dilation=1,
    stride=1,
):
    """
    ELSA operation implementation (works with both CPU and GPU tensors)

    Args:
        features: Input features (B, C, H, W)
        ghost_mul: Ghost multiplicative head (B, C, K, K) or None
        ghost_add: Ghost additive head (B, C, K, K) or None
        h_attn: Hadamard attention weights (B, K*K, H, W)
        lam: Lambda parameter for ghost_mul
        gamma: Gamma parameter for ghost_add
        kernel_size: Attention kernel size
        dilation: Dilation rate
        stride: Stride for output

    Returns:
        Enhanced features with ELSA attention applied
    """
    B, C, H, W = features.shape
    ks = kernel_size
    device = features.device
    dtype = features.dtype

    # Apply lambda and gamma parameters
    if ghost_mul is not None and lam != 0:
        ghost_mul = ghost_mul**lam
    else:
        ghost_mul = torch.ones((B, C, ks, ks), device=device, dtype=dtype)

    if ghost_add is not None and gamma != 0:
        ghost_add = ghost_add * gamma
    else:
        ghost_add = torch.zeros((B, C, ks, ks), device=device, dtype=dtype)

    # Memory-efficient unfold using conv2d groups
    _pad = kernel_size // 2 * dilation

    # Process in smaller chunks to reduce memory usage
    chunk_size = max(1, B // 4)  # Process batch in chunks
    outputs = []

    for i in range(0, B, chunk_size):
        end_idx = min(i + chunk_size, B)
        features_chunk = features[i:end_idx]
        ghost_mul_chunk = ghost_mul[i:end_idx] if ghost_mul is not None else ghost_mul
        ghost_add_chunk = ghost_add[i:end_idx] if ghost_add is not None else ghost_add
        h_attn_chunk = h_attn[i:end_idx]

        B_chunk = features_chunk.shape[0]

        # Unfold features for local attention
        features_unfolded = F.unfold(
            features_chunk,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=_pad,
            stride=stride,
        ).reshape(B_chunk, C, kernel_size**2, H * W)

        # Reshape ghost heads and attention
        ghost_mul_chunk = ghost_mul_chunk.reshape(B_chunk, C, kernel_size**2, 1)
        ghost_add_chunk = ghost_add_chunk.reshape(B_chunk, C, kernel_size**2, 1)
        h_attn_chunk = h_attn_chunk.reshape(B_chunk, 1, kernel_size**2, H * W)

        # Apply ELSA: enhanced filters = ghost_mul * h_attn + ghost_add
        enhanced_filters = (
            ghost_mul_chunk * h_attn_chunk + ghost_add_chunk
        )  # B_chunk, C, K*K, H*W

        # Apply enhanced filters to features
        output_chunk = (features_unfolded * enhanced_filters).sum(2)  # B_chunk, C, H*W
        outputs.append(output_chunk.reshape(B_chunk, C, H // stride, W // stride))

        # Clear intermediate tensors
        del features_unfolded, enhanced_filters, output_chunk

    # Concatenate results
    return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]


class ELSAAttention(nn.Module):
    """
    Enhanced Local Self-Attention (ELSA) module

    This replaces the standard local self-attention with:
    1. Hadamard product for Q-K interaction instead of matrix multiplication
    2. Ghost head mechanism for enhanced spatial attention
    3. Learnable multiplicative and additive ghost parameters
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        group_width=8,
        lam=1.0,
        gamma=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.group_width = group_width
        self.lam = lam  # Ghost multiplicative parameter
        self.gamma = gamma  # Ghost additive parameter

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Adjust dimensions for group convolution
        self.dim_qk = dim // 3 * 2  # 2/3 of dim for Q and K
        if self.dim_qk % group_width != 0:
            self.dim_qk = math.ceil(float(self.dim_qk) / group_width) * group_width

        self.dim_v = dim  # Full dim for V

        print(
            f"ELSA: lambda={lam}, gamma={gamma}, scale={self.scale}, kernel_size={kernel_size}"
        )

        # QKV projection
        self.qkv = nn.Linear(dim, self.dim_qk * 2 + self.dim_v, bias=qkv_bias)

        # Hadamard attention generation network
        self.hadamard_attn = nn.Sequential(
            nn.Conv2d(
                self.dim_qk,
                self.dim_qk,
                kernel_size,
                padding=(kernel_size // 2),
                groups=self.dim_qk // group_width,
            ),
            nn.GELU(),
            nn.Conv2d(self.dim_qk, kernel_size**2 * num_heads, 1),
        )

        # Ghost head parameters for enhanced spatial modeling
        if self.lam != 0 and self.gamma != 0:
            # Both multiplicative and additive ghost heads
            ghost_mul = torch.randn(1, 1, self.dim_v, kernel_size, kernel_size)
            ghost_add = torch.zeros(1, 1, self.dim_v, kernel_size, kernel_size)
            trunc_normal_(ghost_add, std=0.02)
            self.ghost_head = nn.Parameter(
                torch.cat((ghost_mul, ghost_add), dim=0), requires_grad=True
            )
        elif self.lam == 0 and self.gamma != 0:
            # Only additive ghost head
            ghost_add = torch.zeros(1, self.dim_v, kernel_size, kernel_size)
            trunc_normal_(ghost_add, std=0.02)
            self.ghost_head = nn.Parameter(ghost_add, requires_grad=True)
        elif self.lam != 0 and self.gamma == 0:
            # Only multiplicative ghost head
            ghost_mul = torch.randn(1, self.dim_v, kernel_size, kernel_size)
            self.ghost_head = nn.Parameter(ghost_mul, requires_grad=True)
        else:
            # No ghost head
            self.ghost_head = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim_v, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        Forward pass of ELSA attention

        Args:
            x: Input features (B_, N, C) where B_ = num_windows*B, N = window_size^2
            mask: Attention mask (optional)

        Returns:
            Enhanced features with same shape as input
        """
        B_, N, C = x.shape
        window_size = int(math.sqrt(N))  # Assume square windows
        H = W = window_size

        # QKV projection
        qkv = self.qkv(x)  # B_, N, (dim_qk*2 + dim_v)
        q, k, v = torch.split(qkv, (self.dim_qk, self.dim_qk, self.dim_v), dim=-1)

        # Reshape for conv operations: (B_, N, C) -> (B_, C, H, W)
        q = q.transpose(1, 2).reshape(B_, self.dim_qk, H, W)
        k = k.transpose(1, 2).reshape(B_, self.dim_qk, H, W)
        v = v.transpose(1, 2).reshape(B_, self.dim_v, H, W)

        # ELSA's key innovation: Hadamard product instead of matrix multiplication
        hadamard_product = q * k * self.scale  # Element-wise multiplication

        # Generate Hadamard attention weights
        h_attn = self.hadamard_attn(hadamard_product)  # B_, K^2*num_heads, H, W

        # Reshape for multi-head attention
        G = self.num_heads
        v = v.reshape(B_ * G, self.dim_v // G, H, W)
        h_attn = h_attn.reshape(B_ * G, -1, H, W).softmax(
            1
        )  # Softmax over kernel positions
        h_attn = self.attn_drop(h_attn)

        # Apply ghost head if available
        ghost_mul = None
        ghost_add = None
        ks = self.kernel_size

        if self.lam != 0 and self.gamma != 0:
            # Both ghost heads
            gh = self.ghost_head.expand(2, B_, self.dim_v, ks, ks).reshape(
                2, B_ * G, self.dim_v // G, ks, ks
            )
            ghost_mul, ghost_add = gh[0], gh[1]
        elif self.lam == 0 and self.gamma != 0:
            # Only additive ghost head
            ghost_add = self.ghost_head.expand(B_, self.dim_v, ks, ks).reshape(
                B_ * G, self.dim_v // G, ks, ks
            )
        elif self.lam != 0 and self.gamma == 0:
            # Only multiplicative ghost head
            ghost_mul = self.ghost_head.expand(B_, self.dim_v, ks, ks).reshape(
                B_ * G, self.dim_v // G, ks, ks
            )

        # Apply ELSA operation
        enhanced_features = elsa_op(
            v, ghost_mul, ghost_add, h_attn, self.lam, self.gamma, self.kernel_size
        )

        # Reshape back to original format
        enhanced_features = enhanced_features.reshape(B_, self.dim_v, H, W)
        enhanced_features = enhanced_features.permute(0, 2, 3, 1).reshape(
            B_, N, self.dim_v
        )

        # Final projection
        x = self.proj(enhanced_features)
        x = self.proj_drop(x)

        return x
