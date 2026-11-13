"""CPU FastEmbed adapter for indexed children and search queries."""

from threading import Lock

from security_rag.config import Settings
from security_rag.core.application.search import InvalidSearchQuery
from security_rag.core.domain.chunks import EMBEDDING_MAX_TOKENS, ChildChunk
from security_rag.core.domain.vectors import validate_vectors
from security_rag.infrastructure.embeddings.fastembed.tokenizer import ChunkTokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self, settings: Settings, tokenizer: ChunkTokenizer) -> None:
        if settings.embedding_model != MODEL_NAME:
            raise ValueError(f"Only {MODEL_NAME} is supported")
        self.settings = settings
        self.tokenizer = tokenizer
        self._model = None
        self._lock = Lock()

    def embed(self, children: tuple[ChildChunk, ...]) -> list[list[float]]:
        if not children or len(children) > self.settings.ingestion_batch_size:
            raise ValueError("Embedding requires a nonempty bounded batch")
        texts = [child.embedding_text for child in children]
        budget = self.settings.child_max_tokens
        if any(self.tokenizer.count(text) > budget for text in texts):
            raise ValueError(f"Child embedding input exceeds {budget} tokens")
        model = self._get_model()
        vectors = [vector.tolist() for vector in model.embed(texts, batch_size=len(texts))]
        validate_vectors(vectors, len(texts))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        if self.tokenizer.count(query) > EMBEDDING_MAX_TOKENS:
            raise InvalidSearchQuery(f"Query exceeds {EMBEDDING_MAX_TOKENS} embedding tokens")
        vectors = [vector.tolist() for vector in self._get_model().query_embed(query)]
        validate_vectors(vectors, 1)
        return vectors[0]

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                model = TextEmbedding(
                    model_name=self.settings.embedding_model,
                    cache_dir=str(self.settings.embedding_cache_dir),
                    providers=["CPUExecutionProvider"],
                    threads=4,
                )
                runtime = model.model.tokenizer
                if runtime is None:
                    raise ValueError("FastEmbed did not initialize its tokenizer")
                actual = ChunkTokenizer(runtime, self.settings.embedding_model)
                if actual.fingerprint != self.tokenizer.fingerprint:
                    raise ValueError("Chunking tokenizer differs from the FastEmbed tokenizer")
                runtime.no_truncation()
                self._model = model
            return self._model
