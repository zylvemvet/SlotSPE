import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack



class Mlp(nn.Module):
    # two mlp, fc-relu-drop-fc-relu-drop
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.1,
            proj_drop=0.1,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, attn = None):
        return_attn = attn is True
        B, N, C = x.shape
        qkv = rearrange(
            self.qkv(x),
            'b n (three h d) -> three b h n d',
            three=3,
            h=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            assert (mask.sum(dim=1) > 0).all()
            mask_q = mask.unsqueeze(1).unsqueeze(-1)
            mask_k = mask.unsqueeze(1).unsqueeze(2)
            final_mask = mask_q * mask_k
            attn = attn.masked_fill(final_mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = rearrange(attn @ v, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return attn
        else:
            return x

class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mlp_ratio=1.,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.1,
                 proj_drop=0.1,
                 drop_path=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ):
        super(Transformer, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x, mask=None):
        attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def get_attn(self, x, mask=None):
        attn = self.attn(self.norm1(x), mask=mask, attn=True)
        return attn


class IterativeCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.1,
        proj_drop=0.1,
        iters=3,
        static_kv=True  # <-- New option here!
    ):
        super().__init__()
        self.num_heads = num_heads
        self.iters = iters
        self.static_kv = static_kv
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gru1 = nn.GRUCell(dim, dim)
        self.gru2 = nn.GRUCell(dim, dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.norm1a = nn.LayerNorm(dim)
        self.norm1b = nn.LayerNorm(dim)
        self.norm2a = nn.LayerNorm(dim)
        self.norm2b = nn.LayerNorm(dim)

    def forward(self, x1, x2, mask1=None, mask2=None, attn = None):
        # Optional: precompute static k/v
        if self.static_kv:
            k1 = rearrange(self.to_k(x1), 'b n (h d) -> b h n d', h=self.num_heads)
            v1 = rearrange(self.to_v(x1), 'b n (h d) -> b h n d', h=self.num_heads)
            k2 = rearrange(self.to_k(x2), 'b n (h d) -> b h n d', h=self.num_heads)
            v2 = rearrange(self.to_v(x2), 'b n (h d) -> b h n d', h=self.num_heads)

        for _ in range(self.iters):
            x1_prev = x1
            x2_prev = x2

            # Normalize
            x1 = self.norm1a(x1)
            x2 = self.norm2a(x2)

            # Query
            q1 = rearrange(self.to_q(x1), 'b n (h d) -> b h n d', h=self.num_heads)
            q2 = rearrange(self.to_q(x2), 'b n (h d) -> b h n d', h=self.num_heads)

            # Dynamic key/value update if not static
            if not self.static_kv:
                k1 = rearrange(self.to_k(x1), 'b n (h d) -> b h n d', h=self.num_heads)
                v1 = rearrange(self.to_v(x1), 'b n (h d) -> b h n d', h=self.num_heads)
                k2 = rearrange(self.to_k(x2), 'b n (h d) -> b h n d', h=self.num_heads)
                v2 = rearrange(self.to_v(x2), 'b n (h d) -> b h n d', h=self.num_heads)

            # Cross attention
            attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale  # x1 attends to x2
            attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale  # x2 attends to x1

            # Apply masks if present
            if mask1 is not None and mask2 is not None:
                assert (mask1.sum(dim=1) > 0).all()
                assert (mask2.sum(dim=1) > 0).all()

                mask_q1 = mask1.unsqueeze(1).unsqueeze(-1)
                mask_k2 = mask2.unsqueeze(1).unsqueeze(2)
                attn1 = attn1.masked_fill(mask_q1 * mask_k2 == 0, -1e9)

                mask_q2 = mask2.unsqueeze(1).unsqueeze(-1)
                mask_k1 = mask1.unsqueeze(1).unsqueeze(2)
                attn2 = attn2.masked_fill(mask_q2 * mask_k1 == 0, -1e9)

            # Attention softmax
            attn1 = self.attn_drop(F.softmax(attn1, dim=-1))
            attn2 = self.attn_drop(F.softmax(attn2, dim=-1))

            # Weighted sum
            update1 = rearrange(attn1 @ v2, 'b h n d -> b n (h d)')
            update2 = rearrange(attn2 @ v1, 'b h n d -> b n (h d)')

            # Project
            update1 = self.proj_drop(self.proj(update1))
            update2 = self.proj_drop(self.proj(update2))

            # GRU update
            update1, packed_shape1 = pack([update1], '* d')
            update2, packed_shape2 = pack([update2], '* d')
            x1_prev, _ = pack([x1_prev], '* d')
            x2_prev, _ = pack([x2_prev], '* d')

            x1 = self.gru1(update1, x1_prev)
            x2 = self.gru2(update2, x2_prev)

            x1, = unpack(x1, packed_shape1,'* d')
            x2, = unpack(x2, packed_shape2,'* d')

            # MLP + residual
            x1 = x1 + self.mlp1(self.norm1b(x1))
            x2 = x2 + self.mlp2(self.norm2b(x2))

        if attn is not None:
            return attn1, attn2
        return x1, x2


class IterativeCrossAttTransformer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mlp_ratio=1.,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.1,
                 proj_drop=0.1,
                 drop_path=0.1,
                 iters=3,
                 static_kv=True
                 ):
        super(IterativeCrossAttTransformer, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = IterativeCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            iters=iters,
            static_kv=static_kv
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop_path)

    def forward(self, x1, x2, mask1=None, mask2=None):
        attn1, attn2 = self.attn(self.norm1(x1), self.norm1(x2), mask1=mask1, mask2=mask2)
        x1 = x1 + self.drop_path(attn1)
        x2 = x2 + self.drop_path(attn2)
        x1 = x1 + self.drop_path(self.mlp(self.norm2(x1)))
        x2 = x2 + self.drop_path(self.mlp(self.norm2(x2)))
        x = torch.cat([x1, x2], dim=1)
        return x

    def get_attn(self, x1, x2, mask1=None, mask2=None):
        attn1, attn2 = self.attn(self.norm1(x1), self.norm1(x2), mask1=mask1, mask2=mask2, attn = True)
        return attn1, attn2
