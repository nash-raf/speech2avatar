import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # torch.func.jvp uses forward AD, and PyTorch flash/fused SDPA does not
        # support forward-mode AD here. Force the eager attention path.
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rotary_pos_emb=None):
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(batch_size, seq_len, channels)
        out = self.proj(out)
        return self.proj_drop(out)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class SequenceEmbed(nn.Module):
    def __init__(self, dim_w, dim_h, norm_layer=None, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_w, dim_h, bias=bias)
        self.norm = norm_layer(dim_h) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class FMTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    @staticmethod
    def framewise_modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, c, rotary_pos_emb=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(self.framewise_modulate(self.norm1(x), shift_msa, scale_msa), rotary_pos_emb)
        x = x + gate_mlp * self.mlp(self.framewise_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size, dim_w):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, dim_w, bias=True)

    @staticmethod
    def framewise_modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.framewise_modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class FlowMatchingTransformer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        self.num_total_frames = self.num_prev_frames + self.num_frames_for_clip

        self.hidden_size = opt.dim_h
        self.mlp_ratio = opt.mlp_ratio
        self.fmt_depth = opt.fmt_depth
        self.num_heads = opt.num_heads

        self.x_embedder = SequenceEmbed(2 * opt.dim_motion, self.hidden_size)

        head_dim = self.hidden_size // self.num_heads
        self.rotary_emb = RotaryEmbedding(head_dim)

        # Keep original IMTalker-style conditioning path and add the extra iMF scalars into the same AdaLN stream.
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.omega_embedder = TimestepEmbedder(self.hidden_size)
        self.t_min_embedder = TimestepEmbedder(self.hidden_size)
        self.t_max_embedder = TimestepEmbedder(self.hidden_size)
        self.c_embedder = nn.Linear(opt.dim_c, self.hidden_size)

        self.blocks = nn.ModuleList(
            [FMTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.fmt_depth)]
        )
        self.decoder = Decoder(self.hidden_size, self.opt.dim_motion)
        self.v_decoder = Decoder(self.hidden_size, self.opt.dim_motion)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        for embedder in [self.t_embedder, self.omega_embedder, self.t_min_embedder, self.t_max_embedder]:
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for decoder in [self.decoder, self.v_decoder]:
            nn.init.constant_(decoder.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(decoder.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(decoder.linear.weight, 0)
            nn.init.constant_(decoder.linear.bias, 0)

    @staticmethod
    def _cat_prev_current(current, previous):
        if previous is None:
            return current
        return torch.cat([previous, current], dim=1)

    def forward(
        self,
        x,
        a,
        prev_x,
        prev_a,
        ref_x,
        gaze,
        prev_gaze,
        pose,
        prev_pose,
        cam,
        prev_cam,
        h,
        omega,
        t_min,
        t_max,
        return_v=True,
    ):
        x = self._cat_prev_current(x, prev_x)
        a = self._cat_prev_current(a, prev_a)
        pose = self._cat_prev_current(pose, prev_pose)
        cam = self._cat_prev_current(cam, prev_cam)
        gaze = self._cat_prev_current(gaze, prev_gaze)

        ref_x = ref_x[:, None, :].expand(-1, x.shape[1], -1)
        x = torch.cat([ref_x, x], dim=-1)
        x = self.x_embedder(x)

        rotary_pos_emb = self.rotary_emb(x, seq_len=x.shape[1])

        cond = self.c_embedder(a + pose + cam + gaze)
        scalar_cond = self.t_embedder(h)
        scalar_cond = scalar_cond + self.omega_embedder(1 - 1 / omega)
        scalar_cond = scalar_cond + self.t_min_embedder(t_min)
        scalar_cond = scalar_cond + self.t_max_embedder(t_max)
        c = cond + scalar_cond.unsqueeze(1)

        for block in self.blocks:
            x = block(x, c, rotary_pos_emb=rotary_pos_emb)

        u = self.decoder(x, c)
        if not return_v:
            return u

        v = self.v_decoder(x, c)
        return u, v
