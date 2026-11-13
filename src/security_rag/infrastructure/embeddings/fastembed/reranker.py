"""Local cross-encoder inference with explicit pair-length checks."""

from threading import Lock

from security_rag.config import Settings

DEFAULT_PAIR_LIMIT = 512


class CrossEncoder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._lock = Lock()
        self._pair_limit = DEFAULT_PAIR_LIMIT

    def score(self, query: str, texts: list[str]) -> list[float]:
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                model = TextCrossEncoder(
                    self.settings.reranker_model,
                    cache_dir=str(self.settings.embedding_cache_dir),
                    providers=["CPUExecutionProvider"],
                    threads=4,
                )
                tokenizer = model.model.tokenizer
                if tokenizer is None:
                    raise ValueError("Reranker tokenizer was not initialized")
                if tokenizer.truncation:
                    self._pair_limit = min(self._pair_limit, tokenizer.truncation["max_length"])
                tokenizer.no_truncation()
                self._model = model
            tokenizer = self._model.model.tokenizer
            pairs = tokenizer.encode_batch([(query, text) for text in texts])
            if any(len(pair.ids) > self._pair_limit for pair in pairs):
                raise ValueError("Search input exceeds the reranker pair token budget")
            return [float(score) for score in self._model.rerank(query, texts, batch_size=8)]
