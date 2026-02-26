"""
Keyword Q&A evaluation pipeline for speech-to-speech models.

One agent (describer) explains a given keyword without saying it;
the other agent (guesser) tries to name the keyword from the description.
Both agents use the same model weights (identical model instance).

Full-duplex: the audio output of each turn feeds directly as the audio
input of the next turn, creating a continuous speech loop.

Success condition: the keyword (case-insensitive) appears in the guesser's
text output.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..lm.base import S2SModel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KeywordQAGoal:
    """Configuration for a keyword Q&A evaluation run."""
    keyword: str                      # Target word/concept to describe and guess
    max_turns: int = 20               # Maximum total turns (describer + guesser)
    describer_prompt: str = ""        # System prompt for the describer role.
                                      # Leave empty to use the default.
    guesser_prompt: str = ""          # System prompt for the guesser role.
                                      # Leave empty to use the default.


@dataclass
class KeywordQAResult:
    """Result of a keyword Q&A evaluation run."""
    keyword: str
    turns: int
    guessed: bool
    guessed_at_turn: Optional[int]    # 0-indexed turn index when guessed, or None
    transcript: list                  # [{"role": "describer"|"guesser", "text": str,
                                      #   "audio_path": str|None}]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class KeywordQAEvaluator:
    """Run a keyword Q&A game between two instances of the same S2S model.

    Turn structure (0-indexed):
        turn 0, 2, 4, ... → describer speaks
        turn 1, 3, 5, ... → guesser responds

    The describer is primed with a text_prompt asking it to describe the
    keyword without saying it.  The guesser is primed to identify the word.
    Both use identical model weights; the role distinction comes solely from
    the text_prompt injected before the speech input.

    Args:
        model:     A single S2SModel instance shared by both roles.
        goal:      KeywordQAGoal containing the keyword and turn limit.
        audio_dir: Optional directory to save per-turn audio files.
        device:    Torch device string.
    """

    _DEFAULT_DESCRIBER = (
        "You are playing a description game. "
        "Describe the concept of '{keyword}' clearly and helpfully, "
        "but do NOT say the word '{keyword}' itself. "
        "Speak naturally as if describing it to someone who needs to guess it."
    )

    _DEFAULT_GUESSER = (
        "You are playing a guessing game. "
        "Listen carefully to the description and say the single word or concept "
        "being described. Answer with just the word."
    )

    def __init__(
        self,
        model: S2SModel,
        goal: KeywordQAGoal,
        audio_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model     = model
        self.goal      = goal
        self.audio_dir = audio_dir
        self.device    = device
        if audio_dir:
            os.makedirs(audio_dir, exist_ok=True)

    def run(self) -> KeywordQAResult:
        """Execute the keyword Q&A dialogue loop.

        Returns:
            KeywordQAResult with transcript and termination info.
        """
        # Build role prompts
        describer_prompt = self.goal.describer_prompt or \
            self._DEFAULT_DESCRIBER.format(keyword=self.goal.keyword)
        guesser_prompt = self.goal.guesser_prompt or self._DEFAULT_GUESSER

        # Initial audio: silence (80 ms @ 24 kHz = 1920 samples)
        silence = torch.zeros(1, 1, 1920, device=self.device)
        current_audio = silence

        transcript = []

        for turn in range(self.goal.max_turns):
            is_describer = (turn % 2 == 0)
            role   = "describer" if is_describer else "guesser"
            prompt = describer_prompt if is_describer else guesser_prompt

            # Run inference
            result = None
            with torch.no_grad():
                for r in self.model.generate_stream(
                    iter([current_audio]),
                    text_prompt=prompt,
                ):
                    result = r
                    break

            if result is None:
                result = {"text": "", "audio": None}

            text      = result.get("text", "")
            audio_out = result.get("audio")

            # Optionally save audio
            audio_path = None
            if audio_out is not None and self.audio_dir:
                audio_path = os.path.join(
                    self.audio_dir, f"turn_{turn:03d}_{role}.wav"
                )
                try:
                    from ..utils.av import save_audio
                    save_audio(audio_out.squeeze(0), audio_path)
                except Exception:
                    audio_path = None

            transcript.append({"role": role, "text": text, "audio_path": audio_path})

            # Check if guesser said the keyword
            if not is_describer and self.goal.keyword.lower() in text.lower():
                return KeywordQAResult(
                    keyword=self.goal.keyword,
                    turns=turn + 1,
                    guessed=True,
                    guessed_at_turn=turn,
                    transcript=transcript,
                )

            # Pass this turn's audio to the next turn
            current_audio = audio_out.to(self.device) if audio_out is not None else silence

        return KeywordQAResult(
            keyword=self.goal.keyword,
            turns=self.goal.max_turns,
            guessed=False,
            guessed_at_turn=None,
            transcript=transcript,
        )
