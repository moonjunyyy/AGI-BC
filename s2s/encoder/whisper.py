"""
Whisper-style speech encoder.
Mel spectrogram computed with torchaudio; transformer encoder matches Whisper architecture.
Load weights from canonical whisper_encoder.safetensors.
"""
import math
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
    _torchaudio_available = True
except ImportError:
    _torchaudio_available = False


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x: torch.Tensor, mask: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        scale = (C // self.n_head) ** -0.5
        q = self.query(x).view(B, T, self.n_head, -1).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, -1).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, -1).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x


class WhisperEncoder(nn.Module):
    """Whisper-style speech encoder.

    Architecture: 2 Conv1d layers + sinusoidal positional encoding + transformer blocks.
    Input: mel spectrogram [B, n_mels, T_mel]
    Output: encoded features [B, T_enc, n_state]

    Default config matches Whisper Large-v2:
        n_mels=80, n_state=1280, n_head=20, n_layer=32
    """

    N_MELS = 80
    HOP_LENGTH = 160
    N_FFT = 400
    SAMPLE_RATE = 16000

    def __init__(
        self,
        n_mels: int = 80,
        n_state: int = 1280,
        n_head: int = 20,
        n_layer: int = 32,
        n_ctx: int = 1500,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_state = n_state
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = nn.Parameter(self._sinusoids(n_ctx, n_state))
        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = nn.LayerNorm(n_state)

        if _torchaudio_available:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.SAMPLE_RATE,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                n_mels=n_mels,
                f_min=0.0,
                f_max=8000.0,
                center=False,
                norm="slaney",
                mel_scale="slaney",
            )
        else:
            self.mel_transform = None

    @staticmethod
    def _sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> torch.Tensor:
        assert channels % 2 == 0
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert raw audio [B, T] or [B, 1, T] to log-mel [B, n_mels, T_mel]."""
        if self.mel_transform is None:
            raise RuntimeError("torchaudio not available; provide pre-computed mel.")
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        mel = self.mel_transform(audio)
        log_mel = torch.clamp(mel, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        return log_mel

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T_mel]
        Returns:
            [B, T_enc, n_state]
        """
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # [B, T, C]
        T = x.shape[1]
        if T <= self.positional_embedding.shape[0]:
            x = x + self.positional_embedding[:T]
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x

    @classmethod
    def from_safetensors(cls, weights_dir: str, config: dict, device: str = "cuda") -> "WhisperEncoder":
        from safetensors.torch import load_file
        state = load_file(f"{weights_dir}/whisper_encoder.safetensors", device=device)
        model = cls(**config)
        model.load_state_dict(state, strict=True)
        return model.to(device)
