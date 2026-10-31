# AI Security Knowledge Assistant

A local RAG assistant that will answer LLM security questions using retrieved OWASP
documentation, CPU embeddings, PostgreSQL for parents, Qdrant for searchable children,
and a configurable OpenAI-compatible LLM.
Built with Python 3.12+, uv, and FastAPI.

## Status

Phases 1–3 are implemented: document downloading, Markdown-aware parent–child
chunking, persistent PostgreSQL parents and ingestion state, local CPU embeddings,
and retryable Qdrant child indexing. Retrieval, API endpoints, answer generation,
and evaluation are planned.

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
take precedence. Keep secrets in `.env`. LLM credentials are optional through Phase 3.

The settings in `.env.example` configure the download directory, per-request timeout,
retry count, Qdrant URL and collection, embedding model, and OpenAI-compatible LLM.
`DOWNLOAD_RETRIES` is the number of retries after the initial attempt (default 2,
maximum 5). `DOWNLOAD_TIMEOUT_SECONDS` must be positive (default 30).

## Local PostgreSQL and Qdrant

After copying `.env.example` to `.env`, set `POSTGRES_PASSWORD` to a local password
and set `DATABASE_URL` to `postgresql://security_rag:<password>@localhost:5432/security_rag`.
URL-encode special characters in the URL password. A generated hexadecimal password
avoids that encoding step. Keep both values in the ignored `.env`, out of Git.
`POSTGRES_USER` and `POSTGRES_DB` default to `security_rag`; keep the URL consistent
if you change them. PostgreSQL initialization settings apply when its volume is first created.

Start Docker, then run from the project root:

```sh
docker compose up -d --wait
curl --fail http://localhost:6333/readyz
uv run security-rag migrate
```

Compose uses PostgreSQL 17 and Qdrant 1.19.2. Both ports bind only to localhost
(PostgreSQL 5432, Qdrant HTTP 6333); both stores use persistent named volumes.
PostgreSQL has a readiness health check. The Qdrant dashboard is available at
[localhost:6333/dashboard](http://localhost:6333/dashboard).

`migrate` explicitly applies the versioned SQL in `migrations/` transactionally and
records checksums in `schema_migrations`. Repeating it is safe; changing an applied
migration is rejected. Add a new migration for subsequent schema changes.
Importing the application and running other commands never migrate the schema.
The Psycopg 3 pool is bounded by `DATABASE_POOL_SIZE` (default 4, maximum 16);
`DATABASE_TIMEOUT_SECONDS` bounds connection acquisition and SQL statements.

```sh
docker compose down
```

Stopping services preserves their named volumes. Downloading and previewing chunks
require neither database.

## Ingest documents

With both stores running and migrations applied:

```sh
uv run security-rag download
uv run security-rag ingest
uv run security-rag stats
```

Keep the API stopped throughout ingestion, including retries. Resume serving only
after ingestion succeeds. Concurrent reindexing and atomic publication across the
two stores are outside this local MVP's scope.

The command processes one document at a time. It commits that revision's parents and
`processing` state to PostgreSQL before embedding and upserting children in bounded
batches (`INGESTION_BATCH_SIZE`, default 32, maximum 256). FastEmbed loads
`BAAI/bge-small-en-v1.5` once, on CPU, only when embeddings are needed. The first run
downloads model weights into the ignored `EMBEDDING_CACHE_DIR` (default `data/models`).
Only this 384-dimensional model is supported for Phase 3. Chunking and runtime
tokenizers must match; exact embedding inputs are checked without truncation.

PostgreSQL stores each complete parent, including original text, source ranges,
headings, Markdown rendering fields, and revision hashes. Qdrant's single configured
collection stores searchable children with deterministic point IDs, their text,
`parent_id`, and source/revision metadata. Parent text is never copied into child payloads.

A completed revision is skipped only when source/configuration hashes, expected
counts, actual parent IDs and metadata, child IDs and payloads, and persisted parent
references match. Missing or damaged records cause reprocessing. After upserts and
verification, obsolete children are deleted and their absence is verified; only
then are obsolete parents deleted and the new revision marked `completed` in one
PostgreSQL transaction. Old-format child points for the document are cleaned up too.

A failed command exits nonzero and prints a report identifying failed files and
error types. Successfully written parents, children, and durable processing state
remain available. Rerun `ingest` after resolving the failure: deterministic upserts
and cleanup steps can be retried, and completed intact documents are skipped.
The two stores do not share a transaction.

Successful ingestion prints JSON containing processed/skipped/failed document counts,
parent/child counts, child token-length ranges, and embedding, parent-persistence,
and indexing durations. JSON ingestion events go to stderr. `stats` reads persisted
revision status, expected counts, and the Qdrant point count without loading models
or reconstructing parents from Markdown.

`PostgresStore.resolve_parents()` accepts requested parent IDs with document/hash/config
revisions and uses one batched primary-key query. It excludes missing or mismatched
parents and works with a new connection pool after restart, without local source files,
a tokenizer, or a corpus-wide parent cache. Retrieval will use this in Phase 5.

## Planned retrieval pipeline

Phase 4 will embed the question, retrieve up to 20 child candidates from Qdrant,
rerank them with the cross-encoder, and return the top three children from `/search`.

In Phase 5, `/ask` will deduplicate those children's parent IDs and fetch the matching
parents from PostgreSQL in one parameterized batch query. It will preserve the rank
of each parent's highest-ranked child, validate document/configuration revisions,
and build the generation context within the LLM's token budget. A request will load
at most three parents into memory with the selected Top-3 retrieval configuration.
The application will not load the parent corpus into memory at startup.

Missing or mismatched parent records will use the matching child text as fallback.
A PostgreSQL lookup failure will be logged and returned as a service-unavailable
response. Answers will cite only sources included in the generation context.

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
Parents and children remain in memory in the per-document `ChunkedDocument`
returned by the chunker. The `chunk` command is a preview: it writes no chunk files
and does not access PostgreSQL or Qdrant. Use `ingest` to persist parents in
PostgreSQL and child vectors in Qdrant. Future answer generation will fetch matching
parents directly from PostgreSQL.

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
links, and validation without writing chunk files. Ingestion tests use an isolated
in-memory Qdrant collection and a transactional parent-store stand-in. They cover
unchanged skips, damaged records, source/configuration changes, interrupted writes,
invalid persisted references, cleanup ordering, and retry recovery.

Run the optional acceptance checks with the real model tokenizer and local corpus:

```sh
TEST_TOKENIZER_PATH=/absolute/path/to/tokenizer.json uv run pytest -q
```

These checks verify complete-input budgets, source coverage, deterministic output,
and in-memory parent resolution for all ten files, plus oversized structured blocks.
They perform no network calls and are skipped without `TEST_TOKENIZER_PATH`; corpus
cases also require the downloaded Markdown files.

Run PostgreSQL integration tests against the local development database:

```sh
# Use the same local connection URL as DATABASE_URL (do not commit it).
TEST_DATABASE_URL='<local PostgreSQL URL>' uv run pytest -q tests/test_postgres_store.py
# Optional: include checks against a live Qdrant instance.
TEST_QDRANT_URL=http://localhost:6333 uv run pytest -q tests/test_qdrant_live.py
```

PostgreSQL tests create and remove a unique disposable schema for each case, leaving
application tables untouched. They verify migrations and rollback, parent round trips,
Markdown fields, batched lookup from a fresh pool, ingestion-state persistence, and
recovery after partial child writes. Qdrant tests create and remove an isolated test
collection. Both integrations are opt-in; default unit tests require no services.

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
- [x] Phase 3: PostgreSQL parent storage, local embeddings, and idempotent Qdrant child ingestion.
- [ ] Phase 4: Top-3 semantic search and API.
- [ ] Phase 5: Grounded answers and source citations.
- [ ] Phase 6: Evaluation dataset, metrics, tests, and documentation.

Implementation will follow the plan's package structure as each capability is
added. OWASP content must retain its attribution and comply with its source license.
