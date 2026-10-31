# AI Security Knowledge Assistant

A local RAG assistant that will answer LLM security questions using retrieved OWASP
documentation, CPU embeddings, Qdrant, and a configurable OpenAI-compatible LLM.
Built with Python 3.12+, uv, and FastAPI.

## Status

Phases 1 and 2 are implemented: validated settings, local Qdrant through Docker
Compose, document downloading, metadata extraction, and Markdown-aware parent–child
chunking with in-memory parents. Embedding, Qdrant ingestion, retrieval, API endpoints,
answer generation, and evaluation are planned.

The implementation plan is in [specs/implementation-plan.md](specs/implementation-plan.md).

## Development setup

Install Python 3.12+ and uv, then run:

```sh
uv sync
cp .env.example .env
uv run security-rag --help
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Run commands from the project root. Settings load from `.env`; environment variables
take precedence. Keep secrets in `.env`. LLM credentials are optional for Phases 1 and 2.

The settings in `.env.example` configure the download directory, per-request timeout,
retry count, Qdrant URL and collection, embedding model, and OpenAI-compatible LLM.
`DOWNLOAD_RETRIES` is the number of retries after the initial attempt (default 2,
maximum 5). `DOWNLOAD_TIMEOUT_SECONDS` must be positive (default 30).

## Local Qdrant

Start Docker, then run:

```sh
docker compose up -d qdrant
curl --fail http://localhost:6333/readyz
```

Compose pins Qdrant to version 1.19.2, exposes its HTTP interface only on localhost,
and persists storage in a Docker named volume. The dashboard is available at
[localhost:6333/dashboard](http://localhost:6333/dashboard).
Collection creation and indexing are part of Phase 3.

```sh
docker compose down
```

Stopping the service preserves the volume. Document downloading does not require Qdrant.

## Download OWASP documents

```sh
uv run security-rag download
uv run security-rag download --refresh
```

The downloader fetches only the ten explicitly listed Markdown files from the
[canonical 2026 source directory](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/tree/main/2026/final).
It saves the original bytes under `data/owasp/` by default, without cloning the
repository or downloading an archive. Downloaded files are ignored by Git.

Existing files are validated and skipped by default. Use `--refresh` to replace
them. Requests have timeouts and bounded exponential retries for transport errors,
HTTP 408/429/5xx, and invalid response content. Other HTTP errors and redirects fail
immediately. Responses must be HTTP 200, have a Markdown/plain-text content type,
and contain nonempty UTF-8 Markdown with the expected document code, year, title,
and a body. Content length is checked when provided for an uncompressed response.

Each validated download is saved through a temporary file and an atomic replacement.
A failed refresh preserves that file's previous contents. The command exits with
status 1 on failure; already completed files remain available, so the command can be
rerun. Invalid existing files require `--refresh`.

Successful commands print JSON containing downloaded/skipped counts and metadata for
all ten documents: document ID, code, title, filename, version, local path, original
GitHub URL, and SHA-256 of the original bytes. Metadata is derived from the local files
each time, rather than stored in sidecar files. No Git commit SHA is recorded.

## Chunk local documents

After downloading the ten documents, run:

```sh
uv run security-rag chunk
```

The command loads the embedding model's tokenizer on demand. The first run may
download `tokenizer.json` from Hugging Face; it does not download embedding weights.
For offline use, set `TOKENIZER_PATH` to a local copy of the matching
`BAAI/bge-small-en-v1.5` tokenizer. `TOKENIZER_REVISION` selects the Hugging Face
revision when no local path is supplied. Tokenizer contents are included in the
configuration hash, so changed tokenizer artifacts create distinct chunk revisions.

ATX headings define section boundaries relative to the document title, including
titles at `##`. A source-position scanner preserves original whitespace, repeated
heading occurrences, introductory content, and heading-like lines inside fences.
Oversized units use LangChain's token-aware `RecursiveCharacterTextSplitter` with
`split_documents`; source slices, IDs, and parent relationships remain explicit.

Parents contain at most 1,200 tokens including heading context. Children contain at
most 450 tokens for the **complete embedding input**, including the document/section
prefix, restored table headers or code fences, and special tokens. Counting uses the
matching tokenizer with truncation and padding disabled. Fitting Markdown blocks
remain intact; oversized blocks split at natural boundaries, then token offsets.
Original text and source ranges remain separate from restored Markdown syntax.

Parents never overlap or combine unrelated sections. Only children from the same
oversized heading unit and parent can overlap, by up to 60 tokens; overlap shrinks
when required to fit. Every child carries a deterministic UUIDv5 `parent_id`.
Source, parent, child, and canonical configuration hashes distinguish revisions;
unchanged input reproduces the same identifiers.

The command prints JSON with document, parent, and child counts, maximum token
lengths, and hashes. Each document's `parent_chunks` lists parent IDs and texts with
their child IDs and texts nested under `children`.
Parents and children remain in memory in the `ChunkedDocument`
returned by the chunker; the command writes no chunk files. The original Markdown
files remain the source for rebuilding parents with the same configuration and
tokenizer. Future answer generation will reconstruct parents from matching local
sources and use child-text fallback when a matching parent revision is unavailable.
Vector persistence belongs to Phase 3.

## Structure and tests

`config.py` owns settings. `ingestion/manifest.py` defines the source allowlist,
`ingestion/downloader.py` handles HTTP and safe persistence, and `ingestion/loader.py`
validates local Markdown and derives document metadata. The CLI coordinates these
operations. `ingestion/markdown.py` tracks Markdown structure and source positions;
`chunker.py`, `chunk_models.py`, and `tokenizer.py` implement bounded chunks, metadata,
and token counting. Importing the package does not download documents or initialize
models.

Unit tests use temporary directories and mocked HTTP responses; they need neither
network access nor Docker, model downloads, or LLM credentials. They cover settings,
metadata and hashing, the manifest, skip/refresh behavior, retries, invalid/incomplete
downloads, preservation of existing files, and CLI success/failure behavior. Chunking
tests cover relative/repeated headings, introductions, long sections, lists, tables,
fences, Unicode, source ranges, overlap boundaries, hashes, deterministic IDs, parent
links, and validation without writing chunk files.

Run the optional acceptance checks with the real model tokenizer and local corpus:

```sh
TEST_TOKENIZER_PATH=/absolute/path/to/tokenizer.json uv run pytest -q
```

These checks verify complete-input budgets, source coverage, deterministic output,
and in-memory parent resolution for all ten files, plus oversized structured blocks.
They perform no network calls and are skipped without `TEST_TOKENIZER_PATH`; corpus
cases also require the downloaded Markdown files.

## OWASP attribution

The downloaded documents are by the OWASP GenAI Security Project / OWASP Foundation,
from *OWASP Top 10 for Large Language Model Applications*, version 2026.
The [source repository](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10#license)
licenses this material under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The downloader makes no changes to the Markdown. Retain attribution and source/license
links when sharing it; indicate changes and apply the license's ShareAlike terms to
adapted OWASP material.

## Implementation checklist

- [x] Initialize the Python package, dependency lock, and development tooling.
- [x] Save the supplied implementation plan under `specs/`.
- [x] Phase 1: Settings, Docker Compose, document downloads, metadata, and tests.
- [x] Phase 2: Markdown-aware parent–child chunking and tokenizer validation.
- [ ] Phase 3: Local embeddings and idempotent Qdrant ingestion.
- [ ] Phase 4: Top-3 semantic search and API.
- [ ] Phase 5: Grounded answers and source citations.
- [ ] Phase 6: Evaluation dataset, metrics, tests, and documentation.

Implementation will follow the plan's package structure as each capability is
added. OWASP content must retain its attribution and comply with its source license.
