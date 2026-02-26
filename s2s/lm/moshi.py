"""
MoshiModel: Joint LM operating on interleaved text + Mimi audio tokens.
Ported from moshi/models/lm.py with einops replaced and simplified.
"""
import math
import typing as tp
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import S2SModel
from ..streaming.module import StreamingModule, StreamingContainer, State


# ---------------------------------------------------------------------------
# Building blocks (from moshi/modules/transformer.py, einops removed)
# ---------------------------------------------------------------------------

def _rms_norm(x: torch.Tensor, alpha: torch.Tensor, eps: float) -> torch.Tensor:
    var = eps + torch.mean(x.float() ** 2, dim=-1, keepdim=True)
    return (x * (alpha.to(var) * torch.rsqrt(var))).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.alpha, self.eps)


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[2]
        if offset + T > self.cos_cached.shape[2]:
            self._build_cache(offset + T + 512)
        cos = self.cos_cached[:, :, offset:offset + T, :]
        sin = self.sin_cached[:, :, offset:offset + T, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot.to(q.dtype), k_rot.to(k.dtype)


class MoshiAttention(nn.Module):
    """Multi-head attention with GQA and RoPE."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope: RotaryEmbedding):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope = rope
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
    ) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope.apply(q, k, offset=offset)

        # GQA: repeat k/v if needed
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, n_rep, T, self.head_dim).reshape(B, self.num_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, n_rep, T, self.head_dim).reshape(B, self.num_heads, T, self.head_dim)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=(past_kv is None))
        out = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv


class MoshiFFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoshiLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, ffn_dim: int, rope: RotaryEmbedding):
        super().__init__()
        self.attn = MoshiAttention(dim, num_heads, num_kv_heads, rope)
        self.ffn = MoshiFFN(dim, ffn_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
    ) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv, offset=offset)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Moshi LM Model
# ---------------------------------------------------------------------------

class MoshiLMCore(nn.Module):
    """Core transformer for the Moshi joint LM."""

    def __init__(self, config: dict):
        super().__init__()
        dim = config.get("dim", 4096)
        num_heads = config.get("num_heads", 32)
        num_kv_heads = config.get("num_kv_heads", 8)
        num_layers = config.get("num_layers", 32)
        ffn_dim = config.get("ffn_dim", int(dim * 8 / 3))
        max_seq_len = config.get("max_seq_len", 4096)
        n_q = config.get("n_q", 8)
        card = config.get("card", 2048)
        text_card = config.get("text_card", 32000)

        self.dim = dim
        self.n_q = n_q

        self.rope = RotaryEmbedding(dim // num_heads, max_seq_len)
        self.text_emb = nn.Embedding(text_card, dim)
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(n_q)])  # +1 for mask token

        self.transformer = nn.ModuleList([
            MoshiLayer(dim, num_heads, num_kv_heads, ffn_dim, self.rope)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)

        # Output heads
        self.text_head = nn.Linear(dim, text_card, bias=False)
        self.linears = nn.ModuleList([nn.Linear(dim, card, bias=False) for _ in range(n_q)])

        # Depformer (small transformer for parallel codebook prediction)
        dep_dim = config.get("dep_dim", 1024)
        dep_heads = config.get("dep_heads", 16)
        dep_layers = config.get("dep_layers", 6)
        dep_kv_heads = config.get("dep_kv_heads", dep_heads)
        dep_ffn = config.get("dep_ffn", dep_dim * 4)
        dep_rope = RotaryEmbedding(dep_dim // dep_heads)

        self.depformer = nn.ModuleList([
            MoshiLayer(dep_dim, dep_heads, dep_kv_heads, dep_ffn, dep_rope)
            for _ in range(dep_layers)
        ])
        self.dep_norm = RMSNorm(dep_dim)
        self.dep_proj = nn.Linear(dim, dep_dim, bias=False)
        self.dep_linears = nn.ModuleList([nn.Linear(dep_dim, card, bias=False) for _ in range(n_q)])

    def forward(
        self,
        text_tokens: torch.Tensor,  # [B, T]
        audio_codes: torch.Tensor,  # [B, n_q, T]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_logits: [B, T, text_card]
            audio_logits: [B, n_q, T, card]
        """
        B, T = text_tokens.shape
        x = self.text_emb(text_tokens)
        for i, emb in enumerate(self.emb):
            x = x + emb(audio_codes[:, i, :])

        past_kvs = [None] * len(self.transformer)
        for i, layer in enumerate(self.transformer):
            x, past_kvs[i] = layer(x, past_kv=None)
        x = self.norm(x)

        text_logits = self.text_head(x)

        # Depformer for audio codes
        dep_x = self.dep_proj(x)
        dep_past = [None] * len(self.depformer)
        for layer in self.depformer:
            dep_x, _ = layer(dep_x)
        dep_x = self.dep_norm(dep_x)
        audio_logits = torch.stack([lin(dep_x) for lin in self.dep_linears], dim=1)  # [B, n_q, T, card]

        return text_logits, audio_logits


class MoshiModel(S2SModel):
    """Moshi-style joint speech-to-speech model.

    Config keys:
        dim, num_heads, num_kv_heads, num_layers, ffn_dim
        n_q, card, text_card
        dep_dim, dep_heads, dep_layers
        mimi_weights_dir (optional)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config_dict = config
        self.lm = MoshiLMCore(config)
        self._mimi = None

    def _get_mimi(self):
        if self._mimi is None and self.config_dict.get("mimi_weights_dir"):
            from ..codec.mimi import MimiModel
            self._mimi = MimiModel.from_safetensors(
                self.config_dict["mimi_weights_dir"],
                self.config_dict.get("mimi_config", {}),
            )
        return self._mimi

    def generate_stream(
        self,
        audio_frames: Iterator[torch.Tensor],
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ) -> Iterator[dict]:
        frames = list(audio_frames)
        if not frames:
            return
        audio = torch.cat(frames, dim=-1)  # [1, 1, T]

        mimi = self._get_mimi()
        if mimi is None:
            yield {"text": "", "audio": None}
            return

        with torch.no_grad():
            # Encode audio to codes
            codes = mimi.encode(audio)  # [1, n_q, T_codes]
            B, n_q, T = codes.shape

            # BOS tokens
            text_in = torch.zeros(B, 1, dtype=torch.long, device=audio.device)
            gen_codes = []
            gen_text = []

            # Autoregressive generation
            past_text = text_in
            for step in range(min(T + max_new_tokens, T + max_new_tokens)):
                text_logits, audio_logits = self.lm(past_text, codes[:, :, :min(step+1, T)])
                # Sample text token
                if temperature > 0:
                    text_probs = F.softmax(text_logits[:, -1, :] / temperature, dim=-1)
                    next_text = torch.multinomial(text_probs, 1)
                else:
                    next_text = text_logits[:, -1, :].argmax(-1, keepdim=True)
                gen_text.append(next_text)
                past_text = torch.cat([past_text, next_text], dim=1)

                if step < T:
                    continue

                # Sample audio codes
                step_codes = []
                for q in range(n_q):
                    if temperature > 0:
                        audio_probs = F.softmax(audio_logits[:, q, -1, :] / temperature, dim=-1)
                        next_code = torch.multinomial(audio_probs, 1)
                    else:
                        next_code = audio_logits[:, q, -1, :].argmax(-1, keepdim=True)
                    step_codes.append(next_code)
                gen_codes.append(torch.cat(step_codes, dim=-1))  # [1, n_q]

                if len(gen_codes) >= max_new_tokens:
                    break

            # Decode generated codes
            audio_out = None
            if gen_codes:
                gen_codes_t = torch.stack(gen_codes, dim=-1)  # [1, n_q, T']
                audio_out = mimi.decode(gen_codes_t)

        # Simple text decode (just token IDs for now)
        text_str = " ".join(str(t.item()) for t in torch.cat(gen_text, dim=1).squeeze(0))
        yield {"text": text_str, "audio": audio_out}

    def forward(self, batch: dict) -> dict:
        text_tokens = batch.get("text_tokens")  # [B, T]
        audio_codes = batch.get("audio_codes")  # [B, n_q, T]

        if text_tokens is None or audio_codes is None:
            raise ValueError("batch must contain 'text_tokens' and 'audio_codes'")

        text_logits, audio_logits = self.lm(text_tokens[:, :-1], audio_codes[:, :, :-1])

        # Text loss
        text_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.shape[-1]),
            text_tokens[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        # Audio loss
        audio_loss = F.cross_entropy(
            audio_logits.reshape(-1, audio_logits.shape[-1]),
            audio_codes[:, :, 1:].reshape(-1),
            ignore_index=-100,
        )
        loss = text_loss + audio_loss
        return {"loss": loss, "text_loss": text_loss, "audio_loss": audio_loss}

    @classmethod
    def from_safetensors(cls, weights_dir: str, config: dict, device: str = "cuda") -> "MoshiModel":
        from safetensors.torch import load_file
        import os
        fpath = os.path.join(weights_dir, "moshi_lm.safetensors")
        model = cls(config)
        if os.path.exists(fpath):
            state = load_file(fpath, device=device)
            model.lm.load_state_dict(state, strict=False)
        return model.to(device)
