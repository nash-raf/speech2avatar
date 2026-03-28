import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import use_fused_attn


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
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps)
        return x * self.weight


class ScalarEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

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
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.proj(x)


class SwiGLUMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rotary_pos_emb=None):
        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

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
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(bsz, seq_len, dim)
        out = self.out_proj(out)
        return self.proj_drop(out)


class FMTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, rotary_pos_emb=None):
        x = x + self.attn(self.norm1(x), rotary_pos_emb=rotary_pos_emb) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)

    def forward(self, x):
        return self.linear(self.norm(x))


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
        self.aux_head_depth = opt.aux_head_depth
        self.num_heads = opt.num_heads

        if not (0 < self.aux_head_depth < self.fmt_depth):
            raise ValueError("aux_head_depth must be in (0, fmt_depth)")

        self.x_embedder = SequenceEmbed(2 * opt.dim_motion, self.hidden_size)
        self.audio_embedder = SequenceEmbed(opt.dim_c, self.hidden_size)
        self.gaze_embedder = SequenceEmbed(opt.dim_c, self.hidden_size)
        self.pose_embedder = SequenceEmbed(opt.dim_c, self.hidden_size)
        self.cam_embedder = SequenceEmbed(opt.dim_c, self.hidden_size)

        head_dim = self.hidden_size // self.num_heads
        self.rotary_emb = RotaryEmbedding(head_dim)

        self.h_embedder = ScalarEmbedder(self.hidden_size)
        self.omega_embedder = ScalarEmbedder(self.hidden_size)
        self.t_min_embedder = ScalarEmbedder(self.hidden_size)
        self.t_max_embedder = ScalarEmbedder(self.hidden_size)

        self.num_time_tokens = opt.num_time_tokens
        self.num_cfg_tokens = opt.num_cfg_tokens
        self.num_interval_tokens = opt.num_interval_tokens

        token_std = 1.0 / math.sqrt(self.hidden_size)
        self.time_tokens = nn.Parameter(torch.randn(self.num_time_tokens, self.hidden_size) * token_std)
        self.omega_tokens = nn.Parameter(torch.randn(self.num_cfg_tokens, self.hidden_size) * token_std)
        self.t_min_tokens = nn.Parameter(torch.randn(self.num_interval_tokens, self.hidden_size) * token_std)
        self.t_max_tokens = nn.Parameter(torch.randn(self.num_interval_tokens, self.hidden_size) * token_std)
        self.prefix_tokens = (
            self.num_time_tokens + self.num_cfg_tokens + 2 * self.num_interval_tokens
        )

        shared_depth = self.fmt_depth - self.aux_head_depth
        self.shared_blocks = nn.ModuleList(
            [FMTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(shared_depth)]
        )
        self.u_heads = nn.ModuleList(
            [FMTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.aux_head_depth)]
        )
        self.v_heads = nn.ModuleList(
            [FMTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.aux_head_depth)]
        )

        self.u_final = FinalLayer(self.hidden_size, self.opt.dim_motion)
        self.v_final = FinalLayer(self.hidden_size, self.opt.dim_motion)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for block in list(self.shared_blocks) + list(self.u_heads) + list(self.v_heads):
            nn.init.zeros_(block.attn_scale)
            nn.init.zeros_(block.mlp_scale)

        nn.init.zeros_(self.u_final.linear.weight)
        nn.init.zeros_(self.u_final.linear.bias)
        nn.init.zeros_(self.v_final.linear.weight)
        nn.init.zeros_(self.v_final.linear.bias)

    def _build_prefix(self, batch_size, h, omega, t_min, t_max):
        h_embed = self.h_embedder(h)
        omega_embed = self.omega_embedder(1 - 1 / omega)
        t_min_embed = self.t_min_embedder(t_min)
        t_max_embed = self.t_max_embedder(t_max)

        time_tokens = self.time_tokens.unsqueeze(0) + h_embed.unsqueeze(1)
        omega_tokens = self.omega_tokens.unsqueeze(0) + omega_embed.unsqueeze(1)
        t_min_tokens = self.t_min_tokens.unsqueeze(0) + t_min_embed.unsqueeze(1)
        t_max_tokens = self.t_max_tokens.unsqueeze(0) + t_max_embed.unsqueeze(1)

        return torch.cat([omega_tokens, t_min_tokens, t_max_tokens, time_tokens], dim=1)

    def _cat_prev_current(self, current, previous):
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
        gaze = self._cat_prev_current(gaze, prev_gaze)
        pose = self._cat_prev_current(pose, prev_pose)
        cam = self._cat_prev_current(cam, prev_cam)

        ref_x = ref_x[:, None, :].expand(-1, x.shape[1], -1)
        motion_tokens = self.x_embedder(torch.cat([ref_x, x], dim=-1))
        motion_tokens = motion_tokens + self.audio_embedder(a)
        motion_tokens = motion_tokens + self.gaze_embedder(gaze)
        motion_tokens = motion_tokens + self.pose_embedder(pose)
        motion_tokens = motion_tokens + self.cam_embedder(cam)

        prefix = self._build_prefix(x.shape[0], h, omega, t_min, t_max)
        seq = torch.cat([prefix, motion_tokens], dim=1)

        rotary_pos_emb = self.rotary_emb(seq, seq_len=seq.shape[1])

        for block in self.shared_blocks:
            seq = block(seq, rotary_pos_emb=rotary_pos_emb)

        u_seq = seq
        for block in self.u_heads:
            u_seq = block(u_seq, rotary_pos_emb=rotary_pos_emb)
        u = self.u_final(u_seq[:, self.prefix_tokens:])

        if not return_v:
            return u

        v_seq = seq
        for block in self.v_heads:
            v_seq = block(v_seq, rotary_pos_emb=rotary_pos_emb)
        v = self.v_final(v_seq[:, self.prefix_tokens:])
        return u, v
