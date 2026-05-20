# https://gist.github.com/lucidrains/5193d38d1d889681dd42feb847f1f6da
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_3d.py

import torch
from torch import nn
from pdb import set_trace as st

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .vit_with_mask import Transformer


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# https://gist.github.com/lucidrains/213d2be85d67d71147d807737460baf4
class ViTVoxel(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 3
        patch_dim = channels * patch_size ** 3

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
            nn.Dropout(dropout)
        )

    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1 = p, p2 = p, p3 = p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


class ViTTriplane(nn.Module):
    def __init__(self, *, image_size, triplane_size, image_patch_size, triplane_patch_size, num_classes, dim, depth, heads, mlp_dim, patch_embed=False, channels = 3, dim_head = 64,  dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % image_patch_size == 0, 'image dimensions must be divisible by the patch size'

        num_patches = (image_size // image_patch_size) ** 2 * triplane_size # 14*14*3
        # patch_dim = channels * image_patch_size ** 3

        self.patch_size = image_patch_size
        self.triplane_patch_size = triplane_patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.patch_embed = patch_embed
        # if self.patch_embed:
        patch_dim = channels * image_patch_size ** 2 * triplane_patch_size # 1
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

    def forward(self, triplane, mask = None):
        p = self.patch_size
        p_3d = self.triplane_patch_size

        x = rearrange(triplane, 'b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1 = p, p2 = p, p3 = p_3d)

        # if self.patch_embed:
        x = self.patch_to_embedding(x) # B 14*14*4 768

        cls_tokens = self.cls_token.expand(triplane.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x, mask)

        return x[:, 1:]