"""A document as an ordered list of labeled passages, and the complete set of things a reader may say about it."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

START = "This is the start of the paper."
END = "This is the end of the paper."
CANNOT_HELP = "I can only read the paper: a question, an option, the instructions, or a word spelled out."

_QUESTION = re.compile(r"q(\d+)")
_OPTION = re.compile(r"q(\d+)\.([a-z])")
_SYMBOLS = {"/": "slash", "+": "plus", "-": "minus", "=": "equals", ".": "point"}


@dataclass(frozen=True)
class Passage:
    label: str  # "title", "instructions", "q3" for question 3, "q3.b" for option b of question 3
    text: str

    @property
    def spoken(self) -> str:
        """The passage as it's read aloud: a question or an option is introduced by its printed number or letter."""
        if m := _QUESTION.fullmatch(self.label):
            return f"Question {m.group(1)}. {self.text}"
        if m := _OPTION.fullmatch(self.label):
            return f"Option {m.group(2).upper()}. {self.text}"
        return self.text

    @property
    def words(self) -> list[str]:
        """The printed words, without the punctuation around them."""
        return [w for w in (raw.strip(".,;:!?()\"'") for raw in self.text.split()) if w]


def spelling(word: str) -> str:
    """'every' -> 'e, v, e, r, y.'  and  '6/9' -> '6, slash, 9.'"""
    return ", ".join(_SYMBOLS.get(c, c) for c in word.lower()) + "."


@dataclass(frozen=True)
class Document:
    passages: tuple[Passage, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Document":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        passages = tuple(Passage(p["label"], p["text"]) for p in data["passages"])
        labels = [p.label for p in passages]
        if len(set(labels)) != len(labels):
            raise ValueError("every passage needs its own label")
        return cls(passages)

    def to_dict(self) -> dict:
        return {"passages": [{"label": p.label, "text": p.text} for p in self.passages]}

    def index(self, label: str) -> int | None:
        label = label.strip().lower()
        return next((i for i, p in enumerate(self.passages) if p.label == label), None)

    @property
    def overview(self) -> str:
        questions = sum(1 for p in self.passages if _QUESTION.fullmatch(p.label))
        return f"The paper has {questions} questions, in {len(self.passages)} passages."

    def allowed_speech(self) -> frozenset[str]:
        """Everything the reader may ever say: each passage word for word, the spelling of each word, and a few
        fixed replies. The gate replaces anything else with CANNOT_HELP."""
        allowed = {START, END, CANNOT_HELP, self.overview}
        allowed |= {p.spoken for p in self.passages}
        allowed |= {spelling(w) for p in self.passages for w in p.words}
        return frozenset(allowed)
