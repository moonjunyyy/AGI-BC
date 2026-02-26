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
from ..utils.log import get_logger

_log = get_logger("s2s.omni2")

# ---------------------------------------------------------------------------
# Known Qwen2 attention configurations
# hidden_size → (num_attention_heads, num_key_value_heads, head_dim)
# Used to disambiguate shapes that are identical under different head splits.
# ---------------------------------------------------------------------------
_QWEN2_HEAD_CONFIGS: dict[int, tuple[int, int, int]] = {
    896:  (14, 2,  64),   # Qwen2-0.5B
    1536: (12, 2, 128),   # Qwen2-1.5B
    2048: (16, 8, 128),   # Qwen2 (some variants / Omni2 custom)
    3584: (28, 4, 128),   # Qwen2-7B
    7168: (64, 8, 128),   # Qwen2-72B
}


# ---------------------------------------------------------------------------
# Speech projector
# ---------------------------------------------------------------------------

class EncoderProjectorConcat(nn.Module):
    """Concatenate k encoder frames → MLP → LLM hidden dim."""

    def __init__(self, encoder_hidden: int, llm_hidden: int, ds_rate: int = 5):
        super().__init__()
        self.k = ds_rate
        self.linear1 = nn.Linear(encoder_hidden * ds_rate, 2048)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(2048, llm_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        discard = T % self.k
        if discard:
            x = x[:, :-discard, :]
        T = x.shape[1]
        x = x.contiguous().view(B, T // self.k, D * self.k)
        return self.linear2(self.relu(self.linear1(x)))


# ---------------------------------------------------------------------------
# Speech generator
# ---------------------------------------------------------------------------

class LLMSpeechGenerator(nn.Module):
    """Small Qwen2 model that maps LLM hidden states → speech unit tokens."""

    def __init__(
        self,
        lm_hidden: int,
        gen_hidden: int,
        gen_num_layers: int = 4,
        gen_num_heads: int = 8,
        gen_kv_heads: Optional[int] = None,    # None → same as gen_num_heads (MHA)
        gen_intermediate: Optional[int] = None, # None → gen_hidden * 4
        gen_vocab_size: int = 1000,
    ):
        super().__init__()
        from transformers import Qwen2ForCausalLM, Qwen2Config
        gen_config = Qwen2Config(
            hidden_size=gen_hidden,
            num_hidden_layers=gen_num_layers,
            num_attention_heads=gen_num_heads,
            num_key_value_heads=gen_kv_heads if gen_kv_heads is not None else gen_num_heads,
            intermediate_size=gen_intermediate if gen_intermediate is not None else gen_hidden * 4,
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
        tts_inputs = (
            torch.cat([tts_inputs, new_hidden_states], dim=0)
            if tts_inputs is not None
            else new_hidden_states
        )
        if is_finished:
            sep_id = torch.tensor(
                [self.model.config.eos_token_id or 0],
                device=tts_inputs.device, dtype=torch.long,
            )
            tts_inputs = torch.cat([tts_inputs, self.model.get_input_embeddings()(sep_id)], dim=0)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=tts_inputs.unsqueeze(0),
                do_sample=True, temperature=1.0, top_p=1.0, num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.model.config.eos_token_id,
                eos_token_id=self.model.config.eos_token_id,
            )
        generated_tokens = outputs[0]
        generated_units  = " ".join(str(t.item()) for t in generated_tokens)
        gen_embs   = self.model.get_input_embeddings()(generated_tokens)
        tts_inputs = torch.cat([tts_inputs, gen_embs], dim=0)
        return tts_inputs, generated_units


# ---------------------------------------------------------------------------
# Omni2 main model
# ---------------------------------------------------------------------------

class Omni2Model(S2SModel):
    """Qwen2-based speech-to-speech model.

    Pipeline:
        audio → WhisperEncoder → EncoderProjectorConcat → Qwen2 → LLMSpeechGenerator → Mimi

    All architecture hyperparameters are auto-inferred from the safetensors
    checkpoint shapes in from_safetensors(); no manual config dict is required
    for standard checkpoints.
    """

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
        self._tokenizer  = None

        # Speech encoder
        from ..encoder.whisper import WhisperEncoder
        enc_cfg = config.get("encoder_config", {})
        self.speech_encoder = WhisperEncoder(**enc_cfg) if enc_cfg else None

        # Speech projector
        self.speech_projector = EncoderProjectorConcat(
            encoder_hidden=encoder_hidden,
            llm_hidden=lm_hidden,
            ds_rate=encoder_ds_rate,
        )

        # Qwen2 backbone
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
            gen_kv_heads=gen_cfg.get("kv_heads"),
            gen_intermediate=gen_cfg.get("intermediate"),
            gen_vocab_size=gen_cfg.get("vocab_size", 1000),
        )

        self._mimi = None

    # ------------------------------------------------------------------
    # Shape-inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_qwen2_heads(
        hidden: int, q_out: int, k_out: int
    ) -> tuple[int, int, int]:
        """Return (num_heads, kv_heads, head_dim) for a Qwen2 layer.

        Consults _QWEN2_HEAD_CONFIGS first (exact match by hidden size),
        then falls back to trying common head_dim values {128, 64, 256}.
        """
        if hidden in _QWEN2_HEAD_CONFIGS:
            return _QWEN2_HEAD_CONFIGS[hidden]
        for hd in (128, 64, 256, 32):
            if q_out % hd == 0 and k_out % hd == 0:
                return q_out // hd, k_out // hd, hd
        # Last resort: treat full output as one "head"
        return q_out, k_out, 1

    @staticmethod
    def _infer_qwen2_config(state: dict) -> dict:
        """Auto-infer Qwen2ForCausalLM config from safetensors state dict.

        Keys in qwen2_lm.safetensors have the 'model.' prefix stripped by
        convert.py, so they look like:
            lm_head.weight, layers.0.self_attn.q_proj.weight, ...
        """
        inferred: dict = {}

        if "lm_head.weight" in state:
            v, h = state["lm_head.weight"].shape
            inferred["lm_vocab_size"] = v
            inferred["lm_hidden"]     = h

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
        hidden = inferred.get("lm_hidden", 0)

        if hidden and q_key in state and k_key in state:
            q_out, k_out = state[q_key].shape[0], state[k_key].shape[0]
            num_h, kv_h, _ = Omni2Model._resolve_qwen2_heads(hidden, q_out, k_out)
            inferred["lm_num_heads"] = num_h
            inferred["lm_kv_heads"]  = kv_h

        if g_key in state:
            inferred["lm_intermediate"] = state[g_key].shape[0]

        return inferred

    @staticmethod
    def _infer_generator_config(state: dict) -> dict:
        """Auto-infer LLMSpeechGenerator config from safetensors state dict.

        Keys in speech_generator.safetensors have 'speech_generator.' stripped:
            model.model.embed_tokens.weight   ← gen_vocab_size, gen_hidden
            model.model.layers.N.…            ← gen_num_layers, gen_num_heads
            input_proj.0.weight               ← lm_hidden (cross-check)
        """
        inferred: dict = {}

        emb_key = "model.model.embed_tokens.weight"
        if emb_key in state:
            vocab, hidden = state[emb_key].shape
            inferred["vocab_size"] = vocab
            inferred["hidden"]     = hidden

        layer_ids = {
            int(m.group(1))
            for k in state
            if (m := re.match(r"^model\.model\.layers\.(\d+)\.", k))
        }
        if layer_ids:
            inferred["num_layers"] = max(layer_ids) + 1

        q_key = "model.model.layers.0.self_attn.q_proj.weight"
        k_key = "model.model.layers.0.self_attn.k_proj.weight"
        g_key = "model.model.layers.0.mlp.gate_proj.weight"
        hidden = inferred.get("hidden", 0)

        if hidden and q_key in state and k_key in state:
            q_out, k_out = state[q_key].shape[0], state[k_key].shape[0]
            num_h, kv_h, _ = Omni2Model._resolve_qwen2_heads(hidden, q_out, k_out)
            inferred["num_heads"] = num_h
            inferred["kv_heads"]  = kv_h

        if g_key in state:
            inferred["intermediate"] = state[g_key].shape[0]

        return inferred

    @staticmethod
    def _fix_qwen2_keys(state: dict) -> dict:
        """Re-add the 'model.' prefix that convert.py strips from Qwen2 keys.

        convert.py stores:  embed_tokens.weight, layers.0.…, lm_head.weight
        Qwen2ForCausalLM expects: model.embed_tokens.weight, model.layers.0.…, lm_head.weight
        """
        fixed: dict = {}
        for k, v in state.items():
            fixed[k if k.startswith("lm_head.") else "model." + k] = v
        return fixed

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _get_mimi(self):
        if self._mimi is None and self.config_dict.get("mimi_weights_dir"):
            from ..codec.mimi import MimiModel
            self._mimi = MimiModel.from_safetensors(
                self.config_dict["mimi_weights_dir"],
                self.config_dict.get("mimi_config", {}),
            )
        return self._mimi

    def encode_speech(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode raw waveform [B,1,T] @ 16 kHz → [B, T', lm_hidden]."""
        if self.speech_encoder is None:
            raise RuntimeError("speech_encoder not initialized")
        mel   = self.speech_encoder.audio_to_mel(audio.squeeze(1))
        feats = self.speech_encoder(mel)
        return self.speech_projector(feats)

    def generate_stream(
        self,
        audio_frames: Iterator[torch.Tensor],
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        text_prompt: Optional[str] = None,
    ) -> Iterator[dict]:
        """Stream inference, optionally prepending a text prompt.

        Args:
            audio_frames: Iterator of [1, 1, T] waveform chunks.
            text_prompt:  Role instruction prepended as LLM token embeddings.
                          Requires a tokenizer loaded via from_safetensors.
        """
        frames = list(audio_frames)
        if not frames:
            return
        audio = torch.cat(frames, dim=-1)

        speech_embs = self.encode_speech(audio)

        if text_prompt and self._tokenizer is not None:
            tok = self._tokenizer(
                text_prompt, return_tensors="pt", add_special_tokens=True,
            ).input_ids.to(speech_embs.device)
            text_embs  = self.language_model.get_input_embeddings()(tok)
            input_embs = torch.cat([text_embs, speech_embs], dim=1)
        else:
            input_embs = speech_embs

        with torch.no_grad():
            lm_out = self.language_model(
                inputs_embeds=input_embs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = lm_out.hidden_states[-1]
        _, units_str = self.speech_generator.generate_units(
            tts_inputs=None,
            new_hidden_states=hidden_states.squeeze(0),
            new_tokens=torch.argmax(lm_out.logits.squeeze(0), dim=-1),
            is_finished=True,
            max_new_tokens=max_new_tokens,
        )

        audio_out = None
        mimi = self._get_mimi()
        if mimi is not None:
            try:
                unit_ids = [int(u) for u in units_str.split() if u.isdigit()]
                if unit_ids:
                    codes = torch.tensor(
                        unit_ids, device=audio.device
                    ).unsqueeze(0).unsqueeze(0)
                    audio_out = mimi.decode(codes)
            except Exception:
                pass

        yield {"text": units_str, "audio": audio_out}

    def forward(self, batch: dict) -> dict:
        """Training forward pass."""
        speech_embs = self.encode_speech(batch["audio"])
        outputs = self.language_model(
            inputs_embeds=speech_embs,
            labels=batch.get("labels"),
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

        All Qwen2 architecture hyperparameters (hidden size, num layers,
        num heads, vocab size, …) are auto-inferred from checkpoint shapes
        so an empty config dict is usually sufficient.

        Args:
            weights_dir: Directory with {qwen2_lm, speech_generator,
                         whisper_encoder, speech_projector}.safetensors
            config:      Optional overrides; keys take precedence over inferred.
            device:      Target device after loading.
            tp_degree:   Apply tensor parallelism to language_model (> 1).
        """
        from safetensors.torch import load_file
        import os

        # ── 1. Load checkpoints to CPU and infer configs ──────────────────
        qwen2_path = os.path.join(weights_dir, "qwen2_lm.safetensors")
        gen_path   = os.path.join(weights_dir, "speech_generator.safetensors")

        qwen2_state: Optional[dict] = None
        gen_state:   Optional[dict] = None
        inferred_lm:  dict = {}
        inferred_gen: dict = {}

        if os.path.exists(qwen2_path):
            _log.info(f"Inspecting {qwen2_path} for Qwen2 config …")
            qwen2_state = load_file(qwen2_path, device="cpu")
            inferred_lm = cls._infer_qwen2_config(qwen2_state)
            _log.info(
                f"Inferred Qwen2: hidden={inferred_lm.get('lm_hidden')} "
                f"layers={inferred_lm.get('lm_num_layers')} "
                f"heads={inferred_lm.get('lm_num_heads')} "
                f"kv_heads={inferred_lm.get('lm_kv_heads')} "
                f"vocab={inferred_lm.get('lm_vocab_size')}"
            )

        if os.path.exists(gen_path):
            _log.info(f"Inspecting {gen_path} for generator config …")
            gen_state = load_file(gen_path, device="cpu")
            inferred_gen = cls._infer_generator_config(gen_state)
            _log.info(
                f"Inferred generator: hidden={inferred_gen.get('hidden')} "
                f"layers={inferred_gen.get('num_layers')} "
                f"heads={inferred_gen.get('num_heads')} "
                f"kv_heads={inferred_gen.get('kv_heads')} "
                f"vocab={inferred_gen.get('vocab_size')}"
            )

        # ── 2. Merge configs (inferred < explicit) ─────────────────────────
        merged_gen = {**inferred_gen, **config.get("generator_config", {})}
        merged = {**inferred_lm, **{k: v for k, v in config.items() if k != "generator_config"}}
        merged["generator_config"] = merged_gen

        # ── 3. Construct model ─────────────────────────────────────────────
        model = cls(merged)

        # ── 4. Load tokenizer ──────────────────────────────────────────────
        if os.path.exists(os.path.join(weights_dir, "tokenizer.json")):
            try:
                from transformers import AutoTokenizer
                model._tokenizer = AutoTokenizer.from_pretrained(weights_dir)
                _log.info("Tokenizer loaded (text_prompt enabled)")
            except Exception as e:
                _log.warning(f"Tokenizer load failed (text_prompt disabled): {e}")

        # ── 5. Load weights ────────────────────────────────────────────────
        # Qwen2 LM — fix key prefix stripped by convert.py
        if qwen2_state is not None:
            fixed = cls._fix_qwen2_keys(qwen2_state)
            missing, unexpected = model.language_model.load_state_dict(fixed, strict=False)
            _log.info(f"qwen2_lm loaded  missing={len(missing)}  unexpected={len(unexpected)}")
            if missing:
                _log.warning(f"  first 5 missing: {missing[:5]}")
            del qwen2_state, fixed

        # Speech generator — keys already match LLMSpeechGenerator.state_dict()
        if gen_state is not None:
            missing, unexpected = model.speech_generator.load_state_dict(gen_state, strict=False)
            _log.info(f"speech_generator loaded  missing={len(missing)}  unexpected={len(unexpected)}")
            if missing:
                _log.warning(f"  first 5 missing: {missing[:5]}")
            del gen_state

        # Remaining components
        for fname, attr in [
            ("whisper_encoder.safetensors",  "speech_encoder"),
            ("speech_projector.safetensors", "speech_projector"),
        ]:
            fpath = os.path.join(weights_dir, fname)
            if not os.path.exists(fpath):
                continue
            state = load_file(fpath, device="cpu")
            submod = getattr(model, attr, None)
            if submod is not None:
                missing, _ = submod.load_state_dict(state, strict=False)
                _log.info(f"{fname} loaded  missing={len(missing)}")
                if missing:
                    _log.warning(f"  first 5 missing: {missing[:5]}")
            del state

        # ── 6. Tensor parallelism ──────────────────────────────────────────
        if tp_degree > 1:
            from ..utils.tp import shard_model
            model.language_model = shard_model(model.language_model, tp_degree)
            _log.info(f"TP degree={tp_degree} applied to language_model")

        _log.info(f"Omni2Model ready on {device}")
        return model.to(device)
