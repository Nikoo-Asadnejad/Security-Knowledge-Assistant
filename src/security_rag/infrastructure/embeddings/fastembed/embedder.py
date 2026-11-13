"""CPU FastEmbed adapter; model initialization happens only on the first write."""

from security_rag.config import Settings
from security_rag.core.domain.chunks import ChildChunk
from security_rag.core.domain.vectors import validate_vectors
from security_rag.infrastructure.embeddings.fastembed.tokenizer import ChunkTokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self, settings: Settings, tokenizer: ChunkTokenizer) -> None:
        if settings.embedding_model != MODEL_NAME:
            raise ValueError(f"Phase 3 supports only {MODEL_NAME}")
        self.settings = settings
        self.tokenizer = tokenizer
        self._model = None

    def embed(self, children: tuple[ChildChunk, ...]) -> list[list[float]]:
        if not children or len(children) > self.settings.ingestion_batch_size:
            raise ValueError("Embedding requires a nonempty bounded batch")
        texts = [child.embedding_text for child in children]
        if any(self.tokenizer.count(text) > 450 for text in texts):
            raise ValueError("Child embedding input exceeds 450 tokens")
        if self._model is None:
            from fastembed import TextEmbedding

            model = TextEmbedding(
                model_name=self.settings.embedding_model,
                cache_dir=str(self.settings.embedding_cache_dir),
                providers=["CPUExecutionProvider"],
                threads=4,
            )
            # Verify against the runtime tokenizer as well: an offline TOKENIZER_PATH
            # must describe the tokenizer actually used for embedding these inputs.
            runtime = model.model.tokenizer
            if runtime is None:
                raise ValueError("FastEmbed did not initialize its tokenizer")
            actual = ChunkTokenizer(runtime, self.settings.embedding_model)
            if actual.fingerprint != self.tokenizer.fingerprint:
                raise ValueError("Chunking tokenizer differs from the FastEmbed tokenizer")
            runtime.no_truncation()
            self._model = model
        vectors = [vector.tolist() for vector in self._model.embed(texts, batch_size=len(texts))]
        validate_vectors(vectors, len(children))
        return vectors
