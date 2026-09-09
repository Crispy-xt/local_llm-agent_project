"""Character-level tokenizer for the educational TinyLM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CharacterTokenizer:
    """Maps characters to integer IDs and back."""

    stoi: dict[str, int]
    itos: dict[int, str]
    unk_token: str = "<unk>"

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("text must not be empty")
        vocabulary = sorted(set(text))
        if cls.unk_token not in vocabulary:
            vocabulary.insert(0, cls.unk_token)
        stoi = {token: index for index, token in enumerate(vocabulary)}
        itos = {index: token for token, index in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(character, self.unk_id) for character in text]

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self.itos[index] for index in ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]}") from error

    def to_dict(self) -> dict:
        return {
            "stoi": self.stoi,
            "itos": {str(index): token for index, token in self.itos.items()},
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CharacterTokenizer":
        stoi = {str(token): int(index) for token, index in payload["stoi"].items()}
        itos = {int(index): str(token) for index, token in payload["itos"].items()}
        return cls(stoi=stoi, itos=itos, unk_token=payload.get("unk_token", "<unk>"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)