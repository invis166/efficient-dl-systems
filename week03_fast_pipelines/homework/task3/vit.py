"""
The source is ViT from lucidrains implementation
https://github.com/lucidrains/vit-pytorch
"""

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from torch.profiler import record_function


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=255, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        with record_function('feedforward'):
            return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** (-0.5)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        with record_function('self_attn'):
            q, k, v = self.qkv(x).chunk(3, dim=-1)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)

            return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        depth,
        heads,
        dim=255,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),  # maybe move to preprocessing
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dropout=dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.BatchNorm1d(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        with record_function('vit'):
            with record_function('path_embedding'):
                x = self.to_patch_embedding(img)
            b, n, _ = x.shape

            with record_function('cls_token'):
                cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
                x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, : (n + 1)]

            x = self.dropout(x)

            x = self.transformer(x)

            x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

            x = self.to_latent(x)

            output = self.mlp_head(x)
            return output