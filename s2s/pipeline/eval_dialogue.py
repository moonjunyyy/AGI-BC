"""
Dual-agent evaluation pipeline for speech-to-speech dialogue.
"""
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..lm.base import S2SModel


@dataclass
class DialogueGoal:
    """Defines the termination condition for a dual-agent dialogue."""
    description: str
    keywords: list[str] = field(default_factory=list)
    max_turns: int = 20
    judge_model: Optional[str] = None  # HF model ID for LLM judge


@dataclass
class EvalResult:
    """Result of a dual-agent evaluation run."""
    turns: int
    transcript: list[dict]  # [{"role": "A"|"B", "text": str, "audio_path": str|None}]
    done_reason: str  # "keyword" | "llm_judge" | "max_turns"
    goal_achieved: bool


class DualAgentEvaluator:
    """Run a dialogue between two S2S models and evaluate against a goal.

    Args:
        model_a: First S2S model (starts the dialogue).
        model_b: Second S2S model (responds).
        goal: DialogueGoal defining termination criteria.
        audio_dir: Directory to save audio files (optional).
        device: Device string.
    """

    def __init__(
        self,
        model_a: S2SModel,
        model_b: S2SModel,
        goal: DialogueGoal,
        audio_dir: Optional[str] = None,
        device: str = "cuda",
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.goal = goal
        self.audio_dir = audio_dir
        self.device = device
        if audio_dir:
            os.makedirs(audio_dir, exist_ok=True)

    def run(self) -> EvalResult:
        """Execute the dual-agent dialogue loop.

        Returns:
            EvalResult with full transcript and termination info.
        """
        transcript = []
        # Start with a silent audio frame to initialize model A
        frame_size = 1920  # 80ms @ 24kHz
        silence = torch.zeros(1, 1, frame_size, device=self.device)
        current_audio = silence
        current_model = self.model_a
        current_role = "A"
        other_model = self.model_b

        for turn in range(self.goal.max_turns):
            # Generate response
            result = None
            with torch.no_grad():
                for r in current_model.generate_stream(iter([current_audio])):
                    result = r
                    break  # Take first result

            if result is None:
                result = {"text": "", "audio": None}

            text = result.get("text", "")
            audio_out = result.get("audio")

            # Save audio if available
            audio_path = None
            if audio_out is not None and self.audio_dir:
                audio_path = os.path.join(self.audio_dir, f"turn_{turn:03d}_{current_role}.wav")
                try:
                    from ..utils.av import save_audio
                    save_audio(audio_out.squeeze(0), audio_path)
                except Exception:
                    audio_path = None

            transcript.append({"role": current_role, "text": text, "audio_path": audio_path})

            # Check termination
            done, reason = self._check_goal(transcript)
            if done:
                return EvalResult(
                    turns=turn + 1,
                    transcript=transcript,
                    done_reason=reason,
                    goal_achieved=(reason != "max_turns"),
                )

            # Prepare next turn: use generated audio as input to other model
            if audio_out is not None:
                current_audio = audio_out.to(self.device)
            else:
                current_audio = silence

            # Swap models
            current_model, other_model = other_model, current_model
            current_role = "B" if current_role == "A" else "A"

        return EvalResult(
            turns=self.goal.max_turns,
            transcript=transcript,
            done_reason="max_turns",
            goal_achieved=False,
        )

    def _check_goal(self, transcript: list[dict]) -> tuple[bool, str]:
        """Check if the goal has been achieved.

        Checks in order:
        1. Keyword match on latest text turn
        2. LLM judge if judge_model is set
        3. max_turns fallback (handled by caller)

        Returns:
            (done: bool, reason: str)
        """
        if not transcript:
            return False, ""

        latest_text = transcript[-1].get("text", "").lower()

        # 1. Keyword match
        if self.goal.keywords:
            for kw in self.goal.keywords:
                if kw.lower() in latest_text:
                    return True, "keyword"

        # 2. LLM judge
        if self.goal.judge_model:
            try:
                achieved = self._llm_judge(transcript)
                if achieved:
                    return True, "llm_judge"
            except Exception:
                pass

        return False, ""

    def _llm_judge(self, transcript: list[dict]) -> bool:
        """Use an LLM to judge whether the dialogue goal was achieved."""
        try:
            from transformers import pipeline as hf_pipeline
            judge = hf_pipeline("text-generation", model=self.goal.judge_model, device=self.device)
        except Exception:
            return False

        # Build prompt
        dialogue_text = "\n".join(
            f"{t['role']}: {t['text']}" for t in transcript[-6:]  # Last 6 turns
        )
        prompt = (
            f"Goal: {self.goal.description}\n\n"
            f"Dialogue:\n{dialogue_text}\n\n"
            "Has the goal been achieved? Answer only 'yes' or 'no':"
        )
        try:
            out = judge(prompt, max_new_tokens=5, do_sample=False)[0]["generated_text"]
            return "yes" in out.lower()
        except Exception:
            return False
