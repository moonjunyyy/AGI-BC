"""
Omni2Model: Whisper encoder → EncoderProjectorConcat → Qwen2 LM → LLMSpeechGenerator → Mimi decode.
Weights loaded from canonical safetensors files produced by tools/convert.py.
"""
import re
import typing as tp
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import S2SModel


# ---------------------------------------------------------------------------
# Speech projector (from llama_omni2/model/speech_projector/speech_projector.py)
# ---------------------------------------------------------------------------

class EncoderProjectorConcat(nn.Module):
    """Concatenate k encoder frames → MLP → LLM hidden dim."""

    def __init__(self, encoder_hidden: int, llm_hidden: int, ds_rate: int = 5):
        super().__init__()
        self.k = ds_rate
        self.linear1 = nn.Linear(encoder_hidden * ds_rate, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, llm_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        discard = T % self.k
        if discard > 0:
            x = x[:, :-discard, :]
        T = x.shape[1]
        x = x.contiguous().view(B, T // self.k, D * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# ---------------------------------------------------------------------------
# Speech generator (from llama_omni2/model/speech_generator/speech_generator.py)
# ---------------------------------------------------------------------------

class LLMSpeechGenerator(nn.Module):
    """Small Qwen2 model that maps LLM hidden states → speech unit tokens."""

    def __init__(self, lm_hidden: int, gen_hidden: int, gen_num_layers: int = 4, gen_num_heads: int = 8, gen_vocab_size: int = 1000):
        super().__init__()
        from transformers import Qwen2ForCausalLM, Qwen2Config
        gen_config = Qwen2Config(
            hidden_size=gen_hidden,
            num_hidden_layers=gen_num_layers,
            num_attention_heads=gen_num_heads,
            num_key_value_heads=gen_num_heads,
            intermediate_size=gen_hidden * 4,
            vocab_size=gen_vocab_size,
        )
        self.model = Qwen2ForCausalLM(gen_config)
        self.input_proj = nn.Sequential(
            nn.Linear(lm_hidden, lm_hidden * 2),
            nn.ReLU(),
            nn.Linear(lm_hidden * 2, gen_hidden),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * gen_hidden, gen_hidden),
            nn.Sigmoid(),
        )

    def fusion(self, rep: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([rep, emb], dim=-1))
        return rep * gate + emb * (1 - gate)

    def generate_units(
        self,
        tts_inputs: Optional[torch.Tensor],
        new_hidden_states: torch.Tensor,
        new_tokens: torch.Tensor,
        is_finished: bool = False,
        max_new_tokens: int = 50,
    ) -> tp.Tuple[torch.Tensor, str]:
        new_hidden_states = self.input_proj(new_hidden_states)
        new_token_emb = self.model.get_input_embeddings()(new_tokens)
        new_hidden_states = self.fusion(new_hidden_states, new_token_emb)
        if tts_inputs is not None:
            tts_inputs = torch.cat([tts_inputs, new_hidden_states], dim=0)
        else:
            tts_inputs = new_hidden_states
        if is_finished:
            sep_token_id = self.model.config.eos_token_id or 0
            sep_id = torch.tensor([sep_token_id], device=tts_inputs.device, dtype=torch.long)
            sep_emb = self.model.get_input_embeddings()(sep_id)
            tts_inputs = torch.cat([tts_inputs, sep_emb], dim=0)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=tts_inputs.unsqueeze(0),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.model.config.eos_token_id,
                eos_token_id=self.model.config.eos_token_id,
            )
        generated_tokens = outputs[0]
        generated_units = " ".join(str(t.item()) for t in generated_tokens)
        gen_embs = self.model.get_input_embeddings()(generated_tokens)
        tts_inputs = torch.cat([tts_inputs, gen_embs], dim=0)
        return tts_inputs, generated_units


# ---------------------------------------------------------------------------
# Omni2 main model
# ---------------------------------------------------------------------------

class Omni2Model(S2SModel):
    """Qwen2-based speech-to-speech model.

    Pipeline:
        audio → WhisperEncoder → EncoderProjectorConcat → Qwen2 → LLMSpeechGenerator → unit tokens → Mimi decode

    Config keys (all auto-inferred from checkpoint if not provided):
        encoder_hidden: int       (Whisper output dim, default 1280)
        encoder_ds_rate: int      (projector downsample rate, default 5)
        lm_hidden: int            (Qwen2 hidden size)
        lm_vocab_size: int        (Qwen2 vocab size)
        lm_num_layers: int        (Qwen2 num layers)
        lm_num_heads: int         (Qwen2 num attention heads)
        lm_kv_heads: int          (Qwen2 num KV heads, for GQA)
        lm_intermediate: int      (Qwen2 FFN intermediate size)
        mimi_weights_dir: str     (optional, for audio decoding)
    """

    # Standard head dim for all Qwen2 variants
    _QWEN2_HEAD_DIM = 128

    def __init__(self, config: dict):
        super().__init__()
        from transformers import Qwen2ForCausalLM, Qwen2Config

        encoder_hidden  = config.get("encoder_hidden", 1280)
        encoder_ds_rate = config.get("encoder_ds_rate", 5)
        lm_hidden       = config.get("lm_hidden", 1536)
        lm_vocab_size   = config.get("lm_vocab_size", 152064)
        lm_num_layers   = config.get("lm_num_layers", 28)
        lm_num_heads    = config.get("lm_num_heads", 12)

        self.config_dict = config
        self._tokenizer  = None  # loaded by from_safetensors if tokenizer files present

        # Speech encoder (optional; loaded from whisper_encoder.safetensors)
        from ..encoder.whisper import WhisperEncoder
        enc_cfg = config.get("encoder_config", {})
        self.speech_encoder = WhisperEncoder(**enc_cfg) if enc_cfg else None

        # Speech projector
        self.speech_projector = EncoderProjectorConcat(
            encoder_hidden=encoder_hidden,
            llm_hidden=lm_hidden,
            ds_rate=encoder_ds_rate,
        )

        # Qwen2 LM backbone
        qwen2_config = Qwen2Config(
            hidden_size=lm_hidden,
            num_hidden_layers=lm_num_layers,
            num_attention_heads=lm_num_heads,
            num_key_value_heads=config.get("lm_kv_heads", lm_num_heads),
            intermediate_size=config.get("lm_intermediate", lm_hidden * 4),
            vocab_size=lm_vocab_size,
        )
        self.language_model = Qwen2ForCausalLM(qwen2_config)

        # Speech generator
        gen_cfg = config.get("generator_config", {})
        self.speech_generator = LLMSpeechGenerator(
            lm_hidden=lm_hidden,
            gen_hidden=gen_cfg.get("hidden", 512),
            gen_num_layers=gen_cfg.get("num_layers", 4),
            gen_num_heads=gen_cfg.get("num_heads", 8),
            gen_vocab_size=gen_cfg.get("vocab_size", 1000),
        )

        # Mimi codec (lazy-loaded on first use)
        self._mimi = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mimi(self):
        if self._mimi is None and self.config_dict.get("mimi_weights_dir"):
            from ..codec.mimi import MimiModel
            self._mimi = MimiModel.from_safetensors(
                self.config_dict["mimi_weights_dir"],
                self.config_dict.get("mimi_config", {}),
            )
        return self._mimi

    @staticmethod
    def _infer_qwen2_config(state: dict) -> dict:
        """Peek at state dict key shapes to auto-infer Qwen2 config."""
        inferred: dict = {}
        _HD = Omni2Model._QWEN2_HEAD_DIM

        if "lm_head.weight" in state:
            v, h = state["lm_head.weight"].shape
            inferred["lm_vocab_size"] = v
            inferred["lm_hidden"]     = h

        # Count transformer layers
        layer_ids = {
            int(m.group(1))
            for k in state
            if (m := re.match(r"^layers\.(\d+)\.", k))
        }
        if layer_ids:
            inferred["lm_num_layers"] = max(layer_ids) + 1

        q_key = "layers.0.self_attn.q_proj.weight"
        k_key = "layers.0.self_attn.k_proj.weight"
        g_key = "layers.0.mlp.gate_proj.weight"
        if q_key in state:
            inferred["lm_num_heads"] = state[q_key].shape[0] // _HD
        if k_key in state:
            inferred["lm_kv_heads"] = state[k_key].shape[0] // _HD
        if g_key in state:
            inferred["lm_intermediate"] = state[g_key].shape[0]

        return inferred

    @staticmethod
    def _fix_qwen2_keys(state: dict) -> dict:
        """Re-add the 'model.' prefix that convert.py strips.

        convert.py strips 'model.' from Qwen2ForCausalLM keys so that
        qwen2_lm.safetensors contains:
            embed_tokens.weight
            layers.0.self_attn.q_proj.weight
            lm_head.weight          ← no prefix, kept as-is

        Qwen2ForCausalLM.load_state_dict() expects:
            model.embed_tokens.weight
            model.layers.0.self_attn.q_proj.weight
            lm_head.weight          ← no prefix
        """
        fixed: dict = {}
        for k, v in state.items():
            if k.startswith("lm_head."):
                fixed[k] = v
            else:
                fixed["model." + k] = v
        return fixed

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode_speech(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode raw waveform to LLM-compatible embeddings.

        Args:
            audio: [B, 1, T] float32 at 16 kHz
        Returns:
            [B, T', lm_hidden]
        """
        if self.speech_encoder is None:
            raise RuntimeError("speech_encoder not initialized")
        mel  = self.speech_encoder.audio_to_mel(audio.squeeze(1))
        feats = self.speech_encoder(mel)       # [B, T_mel, enc_hidden]
        proj  = self.speech_projector(feats)   # [B, T', lm_hidden]
        return proj

    def generate_stream(
        self,
        audio_frames: Iterator[torch.Tensor],
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        text_prompt: Optional[str] = None,
    ) -> Iterator[dict]:
        """Stream inference: process audio frames, optionally prepend a text prompt.

        Args:
            audio_frames: Iterator of [1, 1, T] waveform chunks.
            temperature:  Sampling temperature.
            max_new_tokens: Max units to generate.
            text_prompt:  Optional text to prepend as LLM context (requires
                          tokenizer to be loaded via from_safetensors).
        """
        frames = list(audio_frames)
        if not frames:
            return
        audio = torch.cat(frames, dim=-1)   # [1, 1, T]

        speech_embs = self.encode_speech(audio)  # [1, T', D]

        # Optionally prepend text-prompt embeddings
        if text_prompt and self._tokenizer is not None:
            tok = self._tokenizer(
                text_prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.to(speech_embs.device)
            text_embs = self.language_model.get_input_embeddings()(tok)  # [1, T_text, D]
            input_embs = torch.cat([text_embs, speech_embs], dim=1)
        else:
            input_embs = speech_embs

        with torch.no_grad():
            lm_out = self.language_model(
                inputs_embeds=input_embs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = lm_out.hidden_states[-1]  # [1, T', D]
        _, units_str = self.speech_generator.generate_units(
            tts_inputs=None,
            new_hidden_states=hidden_states.squeeze(0),
            new_tokens=torch.argmax(lm_out.logits.squeeze(0), dim=-1),
            is_finished=True,
            max_new_tokens=max_new_tokens,
        )

        # Decode speech units → audio via Mimi (optional)
        audio_out = None
        mimi = self._get_mimi()
        if mimi is not None:
            try:
                unit_ids = [int(u) for u in units_str.split() if u.isdigit()]
                if unit_ids:
                    codes = torch.tensor(unit_ids, device=audio.device).unsqueeze(0).unsqueeze(0)
                    audio_out = mimi.decode(codes)
            except Exception:
                audio_out = None

        yield {"text": units_str, "audio": audio_out}

    def forward(self, batch: dict) -> dict:
        """Training forward pass."""
        audio  = batch["audio"]           # [B, 1, T]
        labels = batch.get("labels")      # [B, S]

        speech_embs = self.encode_speech(audio)
        outputs = self.language_model(
            inputs_embeds=speech_embs,
            labels=labels,
            return_dict=True,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    def from_safetensors(
        cls,
        weights_dir: str,
        config: dict,
        device: str = "cuda",
        tp_degree: int = 1,
    ) -> "Omni2Model":
        """Load model from canonical safetensors layout.

        Steps:
          1. Peek at qwen2_lm.safetensors shapes to auto-infer Qwen2 config.
             Explicit keys in `config` always take precedence over inferred values.
          2. Construct model with the merged config.
          3. Load tokenizer if present (enables text_prompt in generate_stream).
          4. Load weights for each component, fixing the key prefix for Qwen2.
          5. Optionally apply tensor parallelism (tp_degree > 1).
        """
        from safetensors.torch import load_file
        import os

        # ── 1. Auto-infer Qwen2 config from checkpoint ────────────────────
        qwen2_path = os.path.join(weights_dir, "qwen2_lm.safetensors")
        qwen2_state: Optional[dict] = None
        inferred: dict = {}

        if os.path.exists(qwen2_path):
            # Load to CPU once; reuse for both shape inspection and weight loading
            qwen2_state = load_file(qwen2_path, device="cpu")
            inferred = cls._infer_qwen2_config(qwen2_state)

        # Explicit config overrides inferred defaults
        merged_config = {**inferred, **config}

        # ── 2. Construct model ────────────────────────────────────────────
        model = cls(merged_config)

        # ── 3. Load tokenizer ─────────────────────────────────────────────
        if os.path.exists(os.path.join(weights_dir, "tokenizer.json")):
            try:
                from transformers import AutoTokenizer
                model._tokenizer = AutoTokenizer.from_pretrained(weights_dir)
            except Exception as e:
                print(f"[omni2] tokenizer load failed (text_prompt disabled): {e}")

        # ── 4. Load weights ───────────────────────────────────────────────
        # Qwen2 LM: fix key prefix stripped by convert.py
        if qwen2_state is not None:
            fixed = cls._fix_qwen2_keys(qwen2_state)
            missing, unexpected = model.language_model.load_state_dict(fixed, strict=False)
            if missing:
                print(f"[omni2] qwen2_lm: {len(missing)} missing keys (first 5: {missing[:5]})")
            del qwen2_state, fixed  # free CPU memory

        # Other components load with identity key mapping
        for fname, attr in [
            ("whisper_encoder.safetensors",  "speech_encoder"),
            ("speech_projector.safetensors", "speech_projector"),
            ("speech_generator.safetensors", "speech_generator"),
        ]:
            fpath = os.path.join(weights_dir, fname)
            if os.path.exists(fpath):
                state = load_file(fpath, device="cpu")
                submodule = getattr(model, attr, None)
                if submodule is not None:
                    missing, _ = submodule.load_state_dict(state, strict=False)
                    if missing:
                        print(f"[omni2] {fname}: {len(missing)} missing keys")
                del state

        # ── 5. Tensor parallelism ─────────────────────────────────────────
        if tp_degree > 1:
            from ..utils.tp import shard_model
            model.language_model = shard_model(model.language_model, tp_degree)
            print(f"[omni2] TP degree={tp_degree} applied to language_model")

        return model.to(device)
