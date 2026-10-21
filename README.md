# AI Security Knowledge Assistant

A local RAG assistant that will answer LLM security questions using retrieved OWASP
documentation, CPU embeddings, Qdrant, and a configurable OpenAI-compatible LLM.
Built with Python 3.12+, uv, and FastAPI.

## Status

Project foundation initialized. Downloading, ingestion, retrieval, API endpoints,
answer generation, and evaluation are planned and are not implemented yet.

The complete supplied plan is preserved in [specs/implementation-plan.md](specs/implementation-plan.md).

## Development setup

Install Python 3.12+ and uv, then run:

```sh
uv sync
cp .env.example .env
uv run security-rag --help
uv run ruff check .
uv run ruff format --check .
```

Environment variables are a template for the planned implementation; the CLI
currently provides help and version information only. Keep secrets in `.env`.

## Implementation checklist

- [x] Initialize the Python package, dependency lock, and development tooling.
- [x] Save the supplied implementation plan under `specs/`.
- [ ] Phase 1: Settings, Docker Compose, document downloads, metadata, and tests.
- [ ] Phase 2: Markdown-aware chunking with tokenizer validation.
- [ ] Phase 3: Local embeddings and idempotent Qdrant ingestion.
- [ ] Phase 4: Top-3 semantic search and API.
- [ ] Phase 5: Grounded answers and source citations.
- [ ] Phase 6: Evaluation dataset, metrics, tests, and documentation.

Implementation will follow the plan's package structure as each capability is
added. OWASP content must retain its attribution and comply with its source license.
