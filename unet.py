from typing import Optional
import torch
from torch import nn, einsum, softmax
import numpy as np

## VARIABLES
EMBEDDING_DIM = 128
NUM_RESNET_BLOCKS = 4
CHANNELS = 3
IMAGE_SIZE = 32
N_ATTENTION_HEADS = 1
N_CHANNELS = 1
NORM_GROUPS = 32
INPUT_CHANNELS = 3
ATTENTION_EVERYWHERE = False
GAMMA_MIN = 13.3
GAMMA_MAX = 5
N_BLOCKS = 32


# Create U-net model
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        attention_parameters = dict(
            n_heads=N_ATTENTION_HEADS,
            n_channels=EMBEDDING_DIM,
            norm_groups=NORM_GROUPS,
        )
        resnet_parameters = dict(
            in_channels=EMBEDDING_DIM,
            out_channels=EMBEDDING_DIM,
            condition_dim=4 * EMBEDDING_DIM,
            norm_num_groups=NORM_GROUPS,
        )

        self.embed_conditioning = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM * 4),
            nn.SiLU(),
            nn.Linear(EMBEDDING_DIM * 4, EMBEDDING_DIM * 4),
            nn.SiLU(),
        )
        total_input_ch = INPUT_CHANNELS
        self.input_conv = nn.Conv2d(total_input_ch, EMBEDDING_DIM, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_parameters),
                attention_block=AttentionBlock(**attention_parameters) if ATTENTION_EVERYWHERE else None,
            )
            for _ in range(N_BLOCKS)
        )

        self.mid_resnet_block_1 = ResnetBlock(**resnet_parameters)
        self.mid_attn_block = AttentionBlock(**attention_parameters)
        self.mid_resnet_block_2 = ResnetBlock(**resnet_parameters)

        resnet_parameters["in_channels"] *= 2  # double input channels due to skip connections
        self.up_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_parameters),
                attention_block=AttentionBlock(**attention_parameters) if ATTENTION_EVERYWHERE else None,
            )
            for _ in range(N_BLOCKS + 1)
        )
        self.output_conv = nn.Sequential(
            nn.GroupNorm(NORM_GROUPS, EMBEDDING_DIM),
            nn.SiLU(),
            zero_init(nn.Conv2d(EMBEDDING_DIM, INPUT_CHANNELS, kernel_size=3, padding=1)),
        )

    def forward(self, z, g_t):
        g_t = g_t.expand(z.shape[0])
        assert g_t.shape == (z.shape[0],)
        t = (g_t - GAMMA_MIN) / (GAMMA_MAX - GAMMA_MIN)
        t_embedding = get_timestep_embedding(t, EMBEDDING_DIM)

        condition = self.embed_conditioning(t_embedding)

        h = self.input_conv(z)
        skip_connections = []
        for down_block in self.down_blocks:
            skip_connections.append(h)
            h = down_block(h, condition)

        skip_connections.append(h)
        h = self.mid_resnet_block_1(h, condition)
        h = self.mid_attn_block(h)
        h = self.mid_resnet_block_2(h, condition)

        for up_block in self.up_blocks:
            h = torch.cat([h, skip_connections.pop()], dim=1)
            h = up_block(h, condition)

        prediction = self.output_conv(h)
        assert prediction.shape == z.shape, (prediction.shape, z.shape)

        # return prediction
        return prediction + z


class ResnetBlock(nn.Module):
    def __init__(self, in_channels=EMBEDDING_DIM, out_channels=None, condition_dim=None, norm_num_groups=32):
        super().__init__()
        out_channels = out_channels or in_channels
        self.out_channels = out_channels
        self.condition_dim = 4 * EMBEDDING_DIM
        self.net1 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.net2 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if condition_dim is not None:
            self.cond_proj = zero_init(nn.Linear(condition_dim, out_channels))
        else:
            self.cond_proj = None

    def forward(self, x, condition):
        h = self.net1(x)
        if self.cond_proj is not None:
            condition = self.cond_proj(condition)
            condition = condition[:, :, None, None]
            h = h + condition

        h = self.net2(h)
        if x.shape[1] != self.out_channels:
            x = self.shortcut_conv(x)
        assert x.shape == h.shape
        return x + h


def attention_inner_heads(qkv, num_heads):
    """Computes attention with heads inside of qkv in the channel dimension.

    Args:
        qkv: Tensor of shape (B, 3*H*C, T) with Qs, Ks, and Vs, where:
            H = number of heads,
            C = number of channels per head.
        num_heads: number of heads.

    Returns:
        Attention output of shape (B, H*C, T).
    """

    bs, width, length = qkv.shape
    ch = width // (3 * num_heads)

    # Split into (q, k, v) of shape (B, H*C, T).
    q, k, v = qkv.chunk(3, dim=1)

    # Rescale q and k. This makes them contiguous in memory.
    scale = ch ** (-1 / 4)  # scale with 4th root = scaling output by sqrt
    q = q * scale
    k = k * scale

    # Reshape qkv to (B*H, C, T).
    new_shape = (bs * num_heads, ch, length)
    q = q.view(*new_shape)
    k = k.view(*new_shape)
    v = v.reshape(*new_shape)

    # Compute attention.
    weight = einsum("bct,bcs->bts", q, k)  # (B*H, T, T)
    weight = softmax(weight.float(), dim=-1).to(weight.dtype)  # (B*H, T, T)
    out = einsum("bts,bcs->bct", weight, v)  # (B*H, C, T)
    return out.reshape(bs, num_heads * ch, length)  # (B, H*C, T)


class Attention(nn.Module):
    """Based on https://github.com/openai/guided-diffusion."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        assert qkv.dim() >= 3, qkv.dim()
        assert qkv.shape[1] % (3 * self.n_heads) == 0
        spatial_dims = qkv.shape[2:]
        qkv = qkv.view(*qkv.shape[:2], -1)  # (B, 3*H*C, T)
        out = attention_inner_heads(qkv, self.n_heads)  # (B, H*C, T)
        return out.view(*out.shape[:2], *spatial_dims)


class AttentionBlock(nn.Module):
    """Self-attention residual block."""

    def __init__(self, n_heads, n_channels, norm_groups):
        super().__init__()
        assert n_channels % n_heads == 0
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels),
            nn.Conv2d(n_channels, 3 * n_channels, kernel_size=1),  # (B, 3 * C, H, W)
            Attention(n_heads),
            zero_init(nn.Conv2d(n_channels, n_channels, kernel_size=1)),
        )

    def forward(self, x):
        return self.layers(x) + x


#### WHAT IT DO?
def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    timesteps *= 1000.0  # In DDPM the time step is in [0, 1000], here [0, 1]
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)


def zero_init(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p.data)
    return module


class UpDownBlock(nn.Module):
    def __init__(self, resnet_block: ResnetBlock, attention_block: Optional[AttentionBlock] = None):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention_block = attention_block

    def forward(self, x, cond):
        x = self.resnet_block(x, cond)
        if self.attention_block is not None:
            x = self.attention_block(x)
        return x
