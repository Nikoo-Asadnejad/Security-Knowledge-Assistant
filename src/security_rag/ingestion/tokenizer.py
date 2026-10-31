"""Model-specific token counting without padding or silent truncation."""

from pathlib import Path

from tokenizers import Tokenizer

from security_rag.ingestion.chunk_models import sha256


class ChunkTokenizer:
    def __init__(self, tokenizer: Tokenizer, model_name: str) -> None:
        # Own a copy: disabling truncation must not mutate an embedding runtime's tokenizer.
        self._tokenizer = Tokenizer.from_str(tokenizer.to_str())
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self.model_name = model_name
        self.fingerprint = sha256(f"{model_name}\n{self._tokenizer.to_str()}")

    @classmethod
    def load(
        cls, model_name: str, *, path: Path | None = None, revision: str = "main"
    ) -> "ChunkTokenizer":
        try:
            tokenizer = (
                Tokenizer.from_file(str(path))
                if path is not None
                else Tokenizer.from_pretrained(model_name, revision=revision)
            )
        except Exception as exc:
            # The Rust tokenizer API raises generic Exceptions for invalid files and
            # download failures. Normalize these at the I/O boundary for CLI callers.
            raise ValueError(f"Could not load tokenizer for {model_name}: {exc}") from exc
        return cls(tokenizer, model_name)

    def count(self, text: str, *, special_tokens: bool = True) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=special_tokens).ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self._tokenizer.encode(text, add_special_tokens=False).offsets
