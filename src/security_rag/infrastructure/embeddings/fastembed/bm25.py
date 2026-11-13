"""FastEmbed BM25 tokenization and weighting."""

from importlib.metadata import version
from pathlib import Path

from security_rag.core.domain.lexical import preprocess
from security_rag.core.domain.sparse_config import SparseConfig
from security_rag.core.domain.vectors import SparseVector, validate_sparse


def tokens(text: str, token_max_length: int) -> list[str]:
    from fastembed.sparse.bm25 import SimpleTokenizer, remove_non_alphanumeric

    result = SimpleTokenizer.tokenize(remove_non_alphanumeric(preprocess(text)))
    if any(len(token) > token_max_length for token in result):
        raise ValueError("Lexical token exceeds BM25_TOKEN_MAX_LENGTH")
    return result


def config_for_average(
    avg_len: float, *, k: float, b: float, token_max_length: int
) -> SparseConfig:
    return SparseConfig(
        fastembed_version=version("fastembed"),
        avg_len=avg_len,
        k=k,
        b=b,
        token_max_length=token_max_length,
    )


class BM25Encoder:
    def __init__(self, config: SparseConfig, cache_dir: Path) -> None:
        from fastembed import SparseTextEmbedding

        self.config = config
        self.model = SparseTextEmbedding(
            model_name=config.model,
            cache_dir=str(cache_dir),
            k=config.k,
            b=config.b,
            avg_len=config.avg_len,
            language=config.language,
            disable_stemmer=config.disable_stemmer,
            stopwords=set(config.stopwords),
            token_max_length=config.token_max_length,
        )

    def encode(self, texts: list[str], *, query: bool = False) -> list[SparseVector]:
        for text in texts:
            if not tokens(text, self.config.token_max_length) and not query:
                raise ValueError("Child contains no lexical terms")
        prepared = [preprocess(text) for text in texts]
        encoded = (
            self.model.query_embed(prepared)
            if query
            else self.model.embed(prepared, batch_size=max(1, len(prepared)))
        )
        vectors = []
        for result in encoded:
            vector = SparseVector(indices=result.indices.tolist(), values=result.values.tolist())
            validate_sparse(vector, allow_empty=query)
            vectors.append(vector)
        if len(vectors) != len(texts):
            raise ValueError("Sparse embedding count does not match input count")
        return vectors
