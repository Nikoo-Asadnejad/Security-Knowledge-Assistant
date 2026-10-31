# AI Security Knowledge Assistant — Implementation Plan

## 1. Project Overview

Build a RAG-powered AI Security Knowledge Assistant using the official OWASP Top 10 for LLM Applications (2026).

The application answers questions about LLM vulnerabilities, security risks, attack scenarios, and mitigations using retrieved OWASP documentation.

**Primary goal:** Learn RAG engineering through practical implementation of document ingestion, Markdown-aware parent–child chunking, dense and BM25 sparse embeddings, PostgreSQL parent storage, Qdrant child indexing, hybrid retrieval, grounded generation, and evaluation.

Keep the project small, modular, maintainable, and suitable for running locally.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Package manager | uv |
| API | FastAPI |
| Validation | Pydantic v2 |
| Vector database | Qdrant (Docker) |
| Parent and ingestion-state database | PostgreSQL (Docker) |
| PostgreSQL client | Psycopg 3 with connection pooling |
| Dense embeddings | FastEmbed — BAAI/bge-small-en-v1.5 |
| Lexical sparse embeddings | FastEmbed — `Qdrant/bm25`, with Qdrant IDF |
| Candidate fusion | Reciprocal rank fusion (RRF) |
| Reranking | Sentence Transformers CrossEncoder — `cross-encoder/ms-marco-MiniLM-L6-v2` |
| LLM | Configurable OpenAI-compatible API |
| Chunking | LangChain text splitters |
| Tokenizer | Embedding-model-compatible tokenizer |
| Reranker runtime | sentence-transformers |
| HTTP client | httpx |
| Testing | pytest |
| Linting | Ruff |
| Settings | pydantic-settings |
| Observability | Python structured logging |

No LangGraph is required for the MVP.

## 3. Data Source and Downloading

Official repository:

https://github.com/GenAI-Security-Project/GenAI-LLM-Top10

Source directory:

`2026/final/`

**Do not clone the repository or download its ZIP archive.**

Instead, download only the following ten Markdown files directly from GitHub:

1. `LLM01_PromptInjection.md`
2. `LLM02_SensitiveInformationDisclosure.md`
3. `LLM03_ExcessiveAgency.md`
4. `LLM04_SupplyChain.md`
5. `LLM05_DataModelPoisoning.md`
6. `LLM06_UnboundedConsumption.md`
7. `LLM07_Misinformation.md`
8. `LLM08_HiddenContextExposure.md`
9. `LLM09_VectorAndEmbeddingWeaknesses.md`
10. `LLM10_ImproperOutputHandling.md`

Raw download base URL:

`https://raw.githubusercontent.com/GenAI-Security-Project/GenAI-LLM-Top10/main/2026/final/`

Download the files using `httpx` and store them locally:

```text
data/
└── owasp/
    ├── LLM01_PromptInjection.md
    ├── LLM02_SensitiveInformationDisclosure.md
    ├── ...
    └── LLM10_ImproperOutputHandling.md
```

### Requirements

- Maintain an explicit manifest of the ten filenames.
- Download only these files.
- Use HTTP timeouts, retries, and response validation.
- Skip downloading existing files by default.
- Provide a `--refresh` option to download updated versions.
- Preserve original Markdown content.
- Validate that each downloaded file is nonempty.
- Extract document ID, document title, and filename.
- Compute SHA-256 hashes of the original content.
- Preserve original GitHub source URLs.
- Keep data downloading separate from application startup.

Example document metadata:

```json
{
  "document_id": "owasp-llm-2026-llm01-prompt-injection",
  "document_code": "LLM01",
  "title": "Prompt Injection",
  "file_name": "LLM01_PromptInjection.md",
  "version": "2026"
}
```

Do not store Git commit SHA metadata.

Preserve OWASP attribution and follow the source's CC BY-SA 4.0 license requirements.

## 4. Chunking Strategy

Implement **Markdown-aware splitting combined with parent–child chunking** as one pipeline. Markdown headings define semantic boundaries; parent–child relationships connect small searchable chunks to larger context units.

Every searchable child chunk must contain a non-null `parent_id` that resolves to the corresponding parent record. The chunker validates this relationship within its per-document output; Phase 3 persists parents in PostgreSQL before storing children in Qdrant. A heading path alone does not provide this relationship.

Do not implement fixed-size chunking as a separate strategy or add a strategy selector.

### Configuration

| Parameter | Value |
|---|---|
| Strategy | Markdown-aware parent–child |
| Parent maximum | 1,200 tokens, including headings |
| Child maximum | 450 tokens for the complete embedding input |
| Overlap between parents or Markdown sections | 0 |
| Child overlap within the same oversized section and parent | Up to 60 tokens |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Dense / sparse prefetch | Up to 20 child hits per branch |
| Reranker candidate pool | 20 child hits |
| Final retrieved children | 3 |

These are the selected MVP defaults, not universally optimal sizes. Keep short sections short rather than padding them to a target size.

### Chunking algorithm

**Step 1 — Parse Markdown structure**

Identify ATX headings (`#` through `######`) and preserve their hierarchy, document title, source positions, and Markdown blocks.

- Determine section levels relative to the document-title heading. The supplied OWASP files use `##` for the title, `###` for sections, and `####` for subsections; do not assume that every title uses `#`.
- Ignore heading-like text inside fenced code blocks.
- Preserve introductory text directly beneath the title, even when it has no section heading.
- Distinguish repeated heading names by their occurrence/source position, not just their text.
- Retain source slices for stored text; do not rely on a splitter that normalizes whitespace to reproduce the original Markdown.

**Step 2 — Build bounded parents**

Start with each section immediately beneath the document title, including its introductory text and nested subsections, as a candidate parent. Treat any document introduction as a separate candidate with the document title as its heading path.

- Keep a candidate of at most 1,200 tokens as one parent.
- Partition an oversized candidate in source order, preferring subsection boundaries, then paragraph, list-item, table-row, code-line, and sentence boundaries as needed.
- Never combine unrelated top-level sections or overlap parents. Preserve all nonempty source content.
- Keep a fitting Markdown block intact. For an oversized block, split at its natural internal boundaries and retain its table-header or code-language context in metadata. Stored `text` remains an original source slice; any table headers or fence markers restored when rendering embedding/generation input count toward the relevant token limit.
- Retain the original heading path and source range on each resulting parent, even when one section produces multiple parents.
- Do not create empty or heading-only parents. A short section may legitimately produce one parent and one child.

Each parent record has an `id`, `document_id`, `section`, `heading_path`, `text`, `parent_index`, `source_start`, `source_end`, `document_hash`, `parent_hash`, and `chunking_config_hash`, plus the document's title, filename, version, and source metadata. Preserve `markdown_prefix` and `markdown_suffix` needed to restore split tables or code fences when rendering context. Source ranges are zero-based, half-open character offsets into the decoded original document; `parent_hash` hashes the parent text. Parents are context records persisted in PostgreSQL during ingestion and are not embedded.

**Step 3 — Split each parent into searchable children**

Within each parent, first split at nested Markdown heading boundaries so a child belongs to one heading path. Keep section introductory text as its own logical unit.

- Keep a logical unit as one child when its complete embedding input fits within 450 tokens.
- Otherwise split recursively, preferring paragraph, list-item, table-row, code-line, and sentence boundaries. Use token-level fallback only when no natural boundary fits.
- Apply up to 60 tokens of overlap only between children split from the same oversized logical unit within the same parent. Reduce overlap when necessary to respect the token limit or ensure forward progress.
- Never overlap across headings or parent boundaries, even if two parents originate from the same large section.
- Preserve lists, code fences, and tables where possible, with the same oversized-block fallback as for parents.
- Avoid empty or heading-only children. Every child belongs to exactly one parent and retains its own full heading path and original source range.

**Step 4 — Add semantic context and validate token budgets**

Include the document title and the child's heading path in the text submitted for embedding, without repeating the title in both fields.

For example:

```text
Document: Prompt Injection
Section: Prevention and Mitigation Strategies

[Actual section content...]
```

Keep the original child content separately in the payload. Store `parent_id` as metadata; do not insert IDs or the full parent text into the embedding input.

Use the embedding model's tokenizer with truncation disabled. The 450-token child limit includes the contextual prefix, child text, any embedding wrapper, and special tokens. Compute the available body budget after accounting for that overhead and validate the exact input before embedding. The model's hard limit is 512 tokens; never rely on silent truncation. Reject a configuration or heading prefix that leaves no usable body budget.

Use the same tokenizer without truncation to enforce the 1,200-token parent limit, including heading context and any restored Markdown syntax. Parents do not have the embedding model's 512-token restriction because they are not embedded. Answer-context budgeting uses the configured generation model's tokenizer.

Use `MarkdownHeaderTextSplitter` for heading metadata where appropriate and a token-aware `RecursiveCharacterTextSplitter` for oversized units. Preserve metadata with `split_documents`; the default character-count length function is insufficient. Keep parent creation, deterministic IDs, and source-text preservation explicit in the ingestion code. Do not add the full LangChain retriever framework solely for this feature.

**Step 5 — Assign identifiers and hashes**

- `document_hash`: SHA-256 of the complete original source file bytes.
- `parent_hash`: SHA-256 of the UTF-8 parent text.
- `chunk_hash`: SHA-256 of the UTF-8 stored child text, excluding the contextual embedding prefix.
- `chunking_config_hash`: SHA-256 of canonical configuration including algorithm version, heading rules, parent/child limits, overlap, embedding model/tokenizer identity, and embedding-prefix format.

Generate deterministic UUIDs with a fixed UUIDv5 namespace:

- Parent `id`: document ID, document hash, configuration hash, section occurrence, and parent position within that section.
- Child `id` / Qdrant point ID: parent ID, child position within the parent, and chunk hash.
- Child `parent_id`: exactly the corresponding parent record's `id`, including for a parent with only one child.

Document revisions intentionally create new IDs. Identical content and configuration must reproduce the same IDs; duplicate headings or repeated text at different positions must not collide.

### Parent lifecycle

The chunker returns parents and children for one document in memory. The `chunk` CLI validates this output and prints statistics without writing chunk files or accessing PostgreSQL or Qdrant. Persistence belongs to the Phase 3 ingestion command.

During ingestion, persist each parent once in PostgreSQL before upserting its children into Qdrant. Process documents incrementally and embeddings in bounded batches, releasing per-document chunk records after ingestion; do not retain the entire corpus in application memory. Keep parent text out of child payloads.

At query time, fetch only the parents referenced by the final reranked children, using a single batch lookup in PostgreSQL. Only resolve a child to a parent with matching IDs, document ID, document hash, and configuration hash. After restart, read persisted parents directly; local Markdown and the chunking tokenizer are needed for ingestion or reprocessing, not parent retrieval. Use the child-text fallback in Section 7 for missing or mismatched records. A corpus-wide parent cache is not required.

### Design references

- [LangChain Markdown splitting](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter): heading metadata, per-section recursive splitting, overlap behavior, and formatting limitations.
- [LangChain parent-document retrieval implementation](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/retrievers/parent_document_retriever.py): searching small children and resolving their stored parent IDs. This is a design reference, not a required dependency.
- [BAAI model card](https://huggingface.co/BAAI/bge-small-en-v1.5): 384-dimensional embeddings and a 512-token input limit.
- [Cross-encoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2): reference implementation and passage-ranking model details.

## 5. PostgreSQL Parent Storage and Qdrant Child Storage

### PostgreSQL parent storage

- Add PostgreSQL to Docker Compose with a persistent named volume and a health check. Keep its development port bound to localhost.
- Configure the connection through `DATABASE_URL`, document it in `.env.example`, and keep credentials out of Git. Use Psycopg 3 with a bounded connection pool and explicit transactions.
- Provide versioned SQL schema migrations and an explicit CLI command to apply them; do not initialize or migrate the schema during API startup.
- Store the complete parent record from Section 4 in `parent_chunks`, with `id` as a UUID primary key. Include original text, heading and source metadata, Markdown rendering fields, and revision hashes so context can be rendered without rereading source files.
- Index `(document_id, document_hash, chunking_config_hash)` for revision checks and cleanup. Use the primary key for batched parent lookups.
- Store ingestion state in `document_ingestions`, keyed by document ID and revision hashes. Record processing/completed status and expected parent/child counts so incomplete revisions cannot be skipped as unchanged.
- Parent IDs are immutable revision-specific identifiers. Changed source or configuration produces new rows; unchanged records can be upserted idempotently.
- Qdrant's `parent_id` is an application-enforced reference to PostgreSQL; a database foreign key cannot enforce it across the two stores.

### Qdrant child storage

Use exactly one Qdrant collection:

`owasp_security_2026`

Embedding model:

`BAAI/bge-small-en-v1.5`

Vector dimensions: 384.

Store only searchable children as vector points. Store parent records in PostgreSQL as described above; do not embed parents or duplicate parent text in Qdrant.

Phase 3 uses an unnamed dense vector with cosine distance. Phase 4 adds a named `bm25` sparse vector to every child point in the same collection, with `SparseVectorParams(modifier=Modifier.IDF)`. Sparse vectors contain term indices and weights; they do not have a fixed 384-dimensional shape or use cosine normalization. Keep the existing dense vector unnamed to avoid an unnecessary dense-vector migration. Sparse and dense vectors share the child's point ID and payload.

Use the pinned Qdrant 1.19.2 server's add-vector-name operation to extend the existing collection, then update only the sparse vector during backfill. Verify this against the real server; do not silently recreate the collection or replace existing dense vectors. New ingestion writes both representations. See [Qdrant collection vector schema updates](https://qdrant.tech/documentation/manage-data/collections/#update-vector-schema).

### Qdrant child point payload

```json
{
  "id": "<deterministic-child-uuid>",
  "parent_id": "<deterministic-parent-uuid>",
  "document_id": "owasp-llm-2026-llm01-prompt-injection",
  "document_code": "LLM01",
  "title": "Prompt Injection",
  "file_name": "LLM01_PromptInjection.md",
  "section": "Prevention and Mitigation Strategies",
  "heading_path": [
    "Prompt Injection",
    "Prevention and Mitigation Strategies"
  ],
  "text": "Actual chunk content...",
  "chunk_index": 0,
  "child_index": 0,
  "source_start": 100,
  "source_end": 500,
  "source_path": "data/owasp/LLM01_PromptInjection.md",
  "source_url": "https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM01_PromptInjection.md",
  "document_hash": "<sha256>",
  "chunk_hash": "<sha256>",
  "chunking_config_hash": "<sha256>",
  "version": "2026"
}
```

From Phase 4, also store `sparse_config_hash` and an `identifiers` keyword array containing recognized codes present in the child's content or heading metadata, including `document_code`. Preserve punctuation in canonical identifier values, for example `CWE-79` and `ERR_CONNECTION_RESET`. These are additive fields; preserve existing source text, revision hashes, IDs, and parent links.

### Point identification

Use the deterministic parent and child UUID rules from Section 4. The Qdrant point ID must equal the payload's child `id`. `chunk_index` is document-wide; `child_index` is relative to the parent. Do not use random IDs for repeated ingestion.

### Idempotent ingestion

- Detect whether the document has already been indexed using PostgreSQL ingestion state.
- Compare both `document_hash` and `chunking_config_hash` with the completed ingestion state.
- Skip documents only when source and configuration are unchanged and the expected parent records in PostgreSQL and child records in Qdrant are complete, with matching IDs and revisions. Counts alone do not establish reference integrity.
- Reprocess changed documents or chunking configurations, including migration from chunks without `parent_id`.
- Validate the per-document parent/child output, then commit the new revision's parents and processing state to PostgreSQL before upserting child points into Qdrant.
- Validate child references against persisted parents with matching document/configuration revisions. Upsert children in bounded batches using deterministic IDs.
- After all new child upserts succeed and stored records are verified, remove obsolete child points from Qdrant. Only after their removal succeeds, delete obsolete parent revisions from PostgreSQL and mark the new revision complete in a transaction. Never delete parents while stored children still reference them.
- Avoid duplicate vectors and stale chunks.
- PostgreSQL and Qdrant do not share a transaction. Preserve incomplete processing state on failure and make every step retryable, including cleanup; never mark a partially indexed document complete. Parent writes that succeeded before a Qdrant failure can be reused on retry.
- Never resolve children to parents from a different source/configuration revision.
- From Phase 4, skipping also requires the expected sparse configuration and valid dense and sparse vectors for every child. Track sparse completion/configuration in PostgreSQL ingestion state through a new versioned migration. A Phase 3 `completed` record alone does not prove sparse readiness. Keep sparse configuration separate from `chunking_config_hash` so a sparse-only change does not regenerate parent/child IDs or dense embeddings. Interrupted backfills remain incomplete and retryable.

For the local MVP, run ingestion while the API is stopped and resume serving only after ingestion completes successfully. Concurrent reindexing and atomic publication across both stores are out of scope.

Keep the Qdrant collection name configurable, but use only one collection for this project.

## 6. Dense and Sparse Embedding Pipeline

Use FastEmbed to generate embeddings locally on the CPU.

The ingestion pipeline:

1. Read locally downloaded Markdown files one document at a time.
2. Parse Markdown headings.
3. Generate bounded Markdown parents and their searchable children.
4. Calculate hashes, deterministic IDs, parent links, and source metadata.
5. Prepare and validate child embedding inputs with contextual headings.
6. Validate parent references and persist parents plus processing state in PostgreSQL.
7. Generate child dense embeddings in bounded batches; from Phase 4, also generate BM25 sparse vectors for those same children. Do not embed parents.
8. Upsert child points into Qdrant, verify both required vector representations, persisted parent references, and payloads, then complete revision cleanup and PostgreSQL ingestion state as specified in Section 5.
9. Log ingestion statistics and release per-document chunk records before processing the next document.

The cross-encoder is loaded and used at retrieval time, not during ingestion.

Track:

- Number of documents processed
- Number of parents and children created
- Child token lengths and invalid parent-reference failures
- Embedding duration
- Sparse encoding duration, nonzero term counts, and sparse backfill/completion status from Phase 4
- Parent persistence duration
- Indexing duration
- Documents skipped as unchanged
- Failed documents

Embedding should be performed offline through the ingestion CLI, not during FastAPI startup.

### BM25 sparse-vector practices (Phase 4)

Use local FastEmbed `SparseTextEmbedding(model_name="Qdrant/bm25")`. Encode children with `embed()` and questions with `query_embed()`; their weighting differs. Let Qdrant apply collection IDF through `Modifier.IDF`; do not multiply IDF into client vectors as well. Use the library's stable term-index mapping, never Python's randomized `hash()`. See the [BM25 model card](https://huggingface.co/Qdrant/bm25) and [FastEmbed implementation](https://github.com/qdrant/fastembed/blob/main/fastembed/sparse/bm25.py).

Apply these project defaults and checks:

- Treat each searchable child as a BM25 document, matching the dense retrieval unit. Construct lexical input from its document code, title, unique heading path, and rendered child body, each once. Exclude UUIDs, hashes, source URLs, and unrelated parent text. Preserve original Markdown separately; do not repeat keywords to simulate field boosting.
- Keep lexical preprocessing in one small shared component for ingestion and queries. Start with case-insensitive matching and `disable_stemmer=True` for this technical corpus, which also disables bundled stopwords. This favors technical terms over English stemming; assess the tradeoff on the labeled questions. Disabling stemming does not disable FastEmbed's punctuation processing.
- Test actual emitted tokens for `LLM01`, `CWE-79`, `CVE-2026-12345`, `ERR_CONNECTION_RESET`, and `0x80070005`. These are tokenizer fixtures, not claims that every code occurs in OWASP. Preserve digits, and explicitly configure and test the token-length limit so long identifiers are not silently discarded.
- For recognized punctuated codes, add one deterministic alphanumeric alias before BM25 encoding on both documents and queries so the complete code can contribute a single term. Preserve the canonical original identifier in `identifiers`; aliases and hashed terms can collide and are not an exact-match guarantee. Keep extraction rules bounded to the supported code patterns rather than building a general code parser.
- BM25 is ranked term matching, not substring, phrase, or Boolean equality search. Create keyword payload indexes for `document_id` and `identifiers`. When a caller explicitly requires an exact identifier, use an exact payload filter on both retrieval branches. Do not hard-filter ordinary questions merely because they mention a code; they may seek related guidance. No separate full-text engine is required.
- Start with BM25 `k=1.2` and `b=0.75`. Measure average lexical child length using the selected analyzer before the initial backfill and freeze it as `avg_len`. FastEmbed's default `256` is a constant, not an automatically computed corpus average. Keep the frozen value for incremental ingestion; changing it requires re-encoding all sparse vectors. Qdrant updates IDF as points change, so corpus updates alone do not require rebuilding all vectors for IDF.
- Hash the sparse model/library version, preprocessing and identifier rules, language/stemming/stopword settings, token limit, lexical-input format, BM25 parameters, and frozen `avg_len` into `sparse_config_hash`. Validate equal index/value lengths, unique nonnegative term indices, finite positive weights, and expected nonempty vectors before writes. Treat a child with no lexical terms as an explicit ingestion failure to resolve, rather than marking it ready.
- Reuse one sparse encoder per process, process bounded batches, and backfill every existing child with the API stopped. Verify vector presence, configuration, payloads, and parent references before marking sparse indexing complete. Support retries and repair of missing or stale sparse vectors without re-embedding unchanged dense content.

Qdrant's current guidance recommends payload full-text BM25 scoring when available; that feature requires v1.20.0, beyond this project's pinned v1.19.2. The selected sparse-vector route meets the requirement to store a sparse vector on every point and fits the existing local FastEmbed setup. A text payload filter alone on this version does not provide ranked BM25. See [Qdrant text indexing and BM25 version requirements](https://qdrant.tech/documentation/manage-data/indexing/#bm25-scoring) and [sparse-vector BM25 guidance](https://qdrant.tech/documentation/search/text-search/sparse-vector-search/).

## 7. Hybrid Retrieval

Implement dense semantic retrieval and BM25 lexical retrieval over the same child points, fuse their rankings with RRF, then apply cross-encoder reranking. Hybrid query execution belongs to Phase 5; Phase 4 prepares sparse indexing only.

Retrieval configuration:

**Dense prefetch-K = 20; sparse prefetch-K = 20; fused candidate-K = 20; final-K = 3.**

The two branches can produce up to 40 unique children before fusion. RRF selects at most 20 for reranking. These are bounded MVP defaults, not guaranteed optimal values.

### Retrieval flow

1. Receive user question.
2. Generate a dense query embedding with the configured dense model and a sparse query vector with the same versioned lexical preprocessing and BM25 encoder used for indexing.
3. Use Qdrant's Query API with two prefetches: up to 20 hits from the existing unnamed dense vector and up to 20 hits using `bm25`. Apply the same optional document or explicit exact-identifier filter within both prefetches, before candidate limits are applied.
4. Fuse with `FusionQuery(fusion=Fusion.RRF)` using the pinned server's default RRF settings, deduplicate by child point ID, and keep up to 20 fused child candidates. Do not add raw dense and BM25 scores: their scales differ. See [Qdrant hybrid queries and RRF](https://qdrant.tech/documentation/search/hybrid-queries/).
5. Score each `(question, child embedding text)` pair with the configured cross-encoder. Run inference in batches, with truncation disabled or explicitly validated against the reranker model's maximum input length.
6. Sort by cross-encoder score descending, then fused rank and child ID for deterministic ties, and retain the final three children, or fewer when fewer are available.
7. Return child IDs, parent IDs, fusion scores, reranker scores, titles, sections, filenames, and source URLs. Report raw branch scores/ranks only when collected; use null for unavailable values, not zero or the fusion score. Qdrant's fused response score is an RRF score, not a dense similarity or a raw BM25 score. Collect branch diagnostics in evaluation/debug runs rather than adding duplicate searches to every request.

The cross-encoder score is the final retrieval rank signal; do not blend it with dense, BM25, or RRF scores. A cross-encoder is query-dependent and cannot be precomputed during ingestion, so load it once per process and reuse it across requests. If reranking fails, log the failure and return up to three fused candidates only when the API contract explicitly marks the response as degraded with null reranker scores; otherwise return service unavailable. Test that reranking does not discard relevant exact-code hits; neither fusion nor reranking guarantees they remain in the final three.

Reject empty questions. A valid question with no lexical terms or no sparse matches may use dense retrieval with an explicit `dense_only` retrieval mode. An encoder/store failure or an incomplete sparse backfill is an error, not an empty lexical match; return service unavailable rather than silently downgrading hybrid retrieval.

For `/ask`, fetch the parents referenced by the final reranked children from PostgreSQL and use them as generation context:

- Deduplicate `parent_id` values while preserving the rank of their highest-scoring matching child, then fetch the corresponding rows in one parameterized batch query through the PostgreSQL parent store. Restore retrieval rank explicitly; database row order is not a ranking signal.
- Validate each fetched parent's ID, document ID, document hash, and configuration hash against its matched children. Load only these parents into request memory; do not rebuild parents from local Markdown or load a corpus-wide parent map at startup.
- Keep matched child IDs and scores for traceability. Scores belong to child matches, not separately scored parents.
- Expand only to the immediate parent; do not recursively fetch the whole document or unrelated siblings.
- Top-3 means the three reranked child hits, which can yield one to three distinct parents. Do not fetch extra hits merely to fill three parents.
- If a parent is missing or its document/configuration revision differs from the child, log an integrity error and fall back to that child's text. Never substitute content from a different revision.
- Handle PostgreSQL lookup failures explicitly: log the failure and return a service-unavailable response. Do not treat an unavailable database as an empty lookup or silently hide the outage behind child-text fallback.
- Enforce the generation model's context budget after reserving space for instructions, the question, formatting, and output. Prefer full parents; when a parent does not fit, use its matching children if they fit. Omit remaining units in rank order rather than silently truncating them, and report insufficient evidence when necessary.

Support optional filtering by `document_id` during both candidate searches. Parent expansion and answer generation are implemented in Phase 6.

Do not implement query rewriting in the MVP.

## 8. RAG Answer Generation

Use a straightforward retrieval-and-generation RAG pipeline:

```text
User Question
    ↓
Dense Query Embedding + BM25 Query Encoding
    ↓
Qdrant Dense + Sparse Prefetch (up to 20 children each)
    ↓
RRF Fusion (up to 20 unique children)
    ↓
Cross-Encoder Reranking (top 20 candidates)
    ↓
Top-3 Reranked Children
    ↓
Deduplicate Parent IDs
    ↓
Batch Fetch Parents from PostgreSQL and Validate Revisions
    ↓
Context Construction
    ↓
LLM Generation
    ↓
Answer + Source Citations
```

### Requirements

- Answer questions using retrieved OWASP content.
- Include citations pointing to the original GitHub documents.
- Include document titles and relevant sections.
- Build context from the deduplicated parents, subject to the context-budget and child-fallback rules in Section 7.
- Preserve parent IDs and matched child IDs in source metadata. Cite only records actually included in the generation context.
- Do not fabricate source references.
- State when evidence is insufficient.
- Treat retrieved documents as untrusted reference material, not executable instructions.
- Keep the LLM client configurable through environment settings.

Do not use LangGraph or agents.

## 9. FastAPI Endpoints

### GET /health

Return service health and PostgreSQL and Qdrant connection status. Report an unavailable required store as unhealthy.

### POST /search

Input:

```json
{
  "query": "What is indirect prompt injection?"
}
```

Return the Top-3 reranked child chunks, with fewer results when fewer are available. `/search` returns the matching child text; `/ask` performs parent expansion internally.

Each result includes:

- Child ID (`id`)
- Parent ID (`parent_id`)
- Text
- RRF fusion score
- Cross-encoder reranker score
- Dense and BM25 branch scores/ranks when available; otherwise null
- Document ID
- Title
- File name
- Section
- Source URL

Optional document filtering can be supported.

Include the retrieval mode (`hybrid` or `dense_only`) and an explicit degraded flag when the documented reranker fallback is used. Exact-identifier filtering, when exposed, must use the canonical `identifiers` payload values; it is separate from ordinary keyword relevance.

Do not expose a collection selector.

### POST /ask

Input:

```json
{
  "question": "How can prompt injection be mitigated?"
}
```

Return:

```json
{
  "answer": "Generated answer...",
  "sources": [
    {
      "document_id": "owasp-llm-2026-llm01-prompt-injection",
      "parent_id": "<deterministic-parent-uuid>",
      "matched_child_ids": ["<deterministic-child-uuid>"],
      "title": "Prompt Injection",
      "file_name": "LLM01_PromptInjection.md",
      "section": "Prevention and Mitigation Strategies",
      "source_url": "https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM01_PromptInjection.md"
    }
  ]
}
```

The retrieval service always uses Top-3.

### CLI commands

Provide commands for:

- Downloading the Markdown documents
- Refreshing downloaded documents
- Applying PostgreSQL schema migrations
- Ingesting parents into PostgreSQL and child vectors into Qdrant
- Backfilling or repairing BM25 sparse vectors on existing child points (Phase 4)
- Printing ingestion statistics
- Running evaluation

Do not expose document ingestion as a public API endpoint.

## 10. Evaluation

Do not implement chunking-size, overlap, or Top-K comparison experiments.

We have selected one configuration:

- Markdown-aware parent–child chunking with cross-encoder reranking
- 1,200-token maximum for parent text including headings
- 450-token maximum for complete child embedding inputs
- Up to 60-token child overlap within the same oversized section and parent
- No overlap between sections or parents
- Prefetch up to 20 dense and 20 BM25 child hits, fuse to 20 candidates with RRF, rerank with the cross-encoder, keep the top 3 children, then expand to parents for answers

Focus evaluation on the quality of this configuration.

### Evaluation dataset

Create approximately 20 manually reviewable test questions covering:

- Vulnerability definitions
- Attack scenarios
- Mitigation strategies
- Differences between security vulnerabilities
- Questions requiring evidence from multiple sections
- Exact OWASP codes and distinctive keywords that are actually present in the corpus
- Mixed code-and-natural-language questions and paraphrases with little lexical overlap
- Unknown identifiers and unsupported questions requiring an insufficient-evidence response

For every question, identify expected relevant documents and sections. Distinguish evidence directly present in retrieved children from evidence added by parent expansion.

Use small isolated fixtures for error codes, CVE/CWE identifiers, punctuation, near-matching codes, repeated terms, and code-like strings missing from the real corpus. Do not insert synthetic errors into the OWASP documents or claim they are source-backed answers. Retrieval can only find codes present in indexed content or metadata.

### Evaluation metrics

Measure:

- Dense-candidate Recall@20
- BM25-candidate Recall@20 and fused-candidate Recall@20
- Reranked Recall@3 and MRR@3 over final child hits, using expected document/section relevance labels
- Exact-code/keyword hit rate at candidate and final stages, including near-match false positives and losses during fusion or reranking
- Relevant-section coverage in the final expanded context after budgeting, reported separately from child retrieval metrics
- Number of distinct parents included and generation-context token count
- Dense and sparse encoding/search latency, fusion latency, reranking latency, and end-to-end retrieval latency
- Embedding duration
- Indexing duration
- Sparse vector coverage, nonzero counts, backfill duration, and sparse configuration identity

Record results as JSON or CSV.

Run dense-only, BM25-only, and hybrid retrieval on the same labels and fixed candidate limits to establish the value of hybrid search, reporting technical-code and semantic question groups separately. This is a focused retrieval baseline check, not a chunk-size or Top-K experiment. Record model/analyzer versions, corpus hashes, and RRF settings. Diagnose each branch independently before changing weights or adding another model.

Add manual answer-quality evaluation for:

- Factual correctness
- Groundedness
- Citation relevance
- Handling of insufficient evidence

Do not introduce an external evaluation framework for the MVP.

## 11. Project Structure

```text
ai-security-rag/
├── src/
│   └── security_rag/
│       ├── api/
│       │   ├── main.py
│       │   ├── routes.py
│       │   └── schemas.py
│       ├── ingestion/
│       │   ├── downloader.py
│       │   ├── loader.py
│       │   ├── chunker.py
│       │   └── pipeline.py
│       ├── embeddings/
│       │   ├── embedder.py
│       │   └── sparse_embedder.py
│       ├── vector_store/
│       │   └── qdrant_store.py
│       ├── parent_store/
│       │   └── postgres_store.py
│       ├── rag/
│       │   ├── retriever.py
│       │   ├── reranker.py
│       │   ├── generator.py
│       │   └── service.py
│       ├── evaluation/
│       │   ├── runner.py
│       │   └── metrics.py
│       ├── config.py
│       └── cli.py
├── data/
│   └── owasp/
├── evaluation/
│   └── questions.json
├── tests/
├── migrations/
│   └── 001_parent_storage.sql
├── compose.yaml
├── pyproject.toml
├── uv.lock
├── .env.example
└── README.md
```

Avoid unnecessary layers, patterns, and abstractions.

## 12. Implementation Phases

### Phase 1 — Project setup and document downloading

- Initialize project with uv.
- Configure dependencies and settings.
- Set up Qdrant using Docker Compose.
- Implement GitHub Markdown downloader.
- Download the ten specified Markdown files.
- Implement metadata extraction and document hashing.
- Write unit tests.

**Acceptance:** All ten files are locally available with valid metadata and hashes, without cloning the repository.

### Phase 2 — Markdown-aware parent–child chunking

- Implement heading-aware parsing.
- Build bounded parents up to 1,200 tokens and children with non-null `parent_id` references.
- Configure a 450-token maximum for complete child embedding inputs.
- Implement conditional overlap of up to 60 tokens within an oversized section and parent.
- Add document title and heading context to embedding text.
- Implement parent/child hashing, configuration hashing, and deterministic parent/child IDs.
- Return per-document parent records in memory without saving chunk files; database persistence is introduced in Phase 3.
- Validate chunk sizes using the embedding tokenizer.
- Test relative heading levels, repeated headings, introductory text, short/long sections, oversized parents, lists, tables, and code fences.
- Test that every child resolves to the correct parent and that overlap never crosses parent or heading boundaries.

**Acceptance:** Every child preserves useful semantic context, respects the complete-input token budget, and resolves to one in-memory parent from the same document/configuration revision. Parents preserve source content within the parent token limit. The chunk command writes no parent or child files.

### Phase 3 — Embeddings, PostgreSQL parents, and Qdrant children

- Add PostgreSQL to Docker Compose with persistent storage, health checks, `DATABASE_URL` settings, and documented setup.
- Add Psycopg 3, bounded connection pooling, and versioned SQL schema migrations with an explicit migration CLI command.
- Implement the PostgreSQL parent store and durable ingestion state, preserving complete parent metadata and revision hashes.
- Provide batched parent lookup by ID for the later retrieval pipeline, with matching-revision validation and no corpus-wide parent loading.
- Initialize FastEmbed.
- Process one document at a time, persist its parents in PostgreSQL, and generate child embeddings locally in bounded batches.
- Create the Qdrant collection.
- Store child vectors, text, parent IDs, and source/revision metadata in Qdrant.
- Implement deterministic point IDs.
- Implement incremental, idempotent ingestion with persisted parent-reference validation, recovery from partial writes across both stores, obsolete-child cleanup before obsolete-parent deletion, and completion tracking.
- Verify PostgreSQL parent records, Qdrant payloads, and persistence across application restarts.

**Acceptance:** All ten OWASP documents have parents persisted in PostgreSQL and searchable children indexed in Qdrant without duplicates or dangling parent references. Stored child points contain parent IDs, titles, filenames, hashes, and source links. Matching parents can be fetched in a batch after an application restart without source reconstruction or loading all parents into memory. Unchanged complete revisions are skipped safely, and retries recover from partial PostgreSQL/Qdrant writes and cleanup failures.

### Phase 4 — BM25 sparse vectors and existing-point backfill

- Implement the local BM25 encoder and shared lexical preprocessing described in Section 6; test identifier tokens and canonical exact-match metadata.
- Measure and freeze lexical average child length, version the sparse configuration, and extend PostgreSQL ingestion state through a new migration without changing chunk IDs or parent revisions.
- Add the named `bm25` sparse vector with Qdrant IDF to the existing single collection, preserving unnamed dense vectors, point IDs, payloads, and parent references.
- Provide an explicit offline CLI operation to backfill/repair sparse vectors on all existing children in bounded batches. Add sparse generation to subsequent ingestion.
- Validate dense/sparse vector content and sparse configuration before skipping or completing a document. Recover from interrupted backfills and repair missing/stale sparse representations without regenerating intact dense vectors.
- Add keyword payload indexes for `document_id` and `identifiers`; preserve original code values for explicit exact filtering.
- Test real-server schema extension and backfill, retry/idempotency, configuration changes, vector coverage, and preservation of dense vectors and parent links. Leave hybrid query orchestration to Phase 5.

**Acceptance:** Every searchable child point has its original dense vector plus a valid `bm25` sparse vector and the expected sparse configuration. Backfill is bounded, retryable, and safe for existing IDs and parent references. Repeated ingestion skips only fully verified dual-vector records. Tokenizer fixtures distinguish supported codes from near matches through canonical identifier metadata, and sparse readiness survives restart.

### Phase 5 — Hybrid retrieval and Search API

- Implement dense and BM25 query encoding and filtered prefetches against the same child points.
- Retrieve up to 20 candidates per branch, fuse to up to 20 unique children with RRF, and rerank them with the cross-encoder in batches.
- Implement `/search`.
- Return the top three reranked children with fusion/reranker scores, correctly labeled optional branch diagnostics, retrieval mode, child/parent IDs, and source metadata.
- Apply optional document and explicit exact-identifier filters consistently to both prefetches.
- Handle empty lexical matches, sparse readiness, branch failures, and reranker degradation explicitly.
- Add semantic, exact-code, keyword, fusion/deduplication, filtering, ranking, and failure-path tests.

**Acceptance:** `/search` runs the hybrid pipeline with at most 20 reranker inputs and three final children. Labeled code/keyword fixtures exercise the sparse branch and near-match handling; semantic paraphrases exercise the dense branch. Dense-only, BM25-only, fused, and reranked results are inspectable on a fixed validation set. Score provenance and degraded behavior are explicit, and relevance changes are measured rather than assumed.

### Phase 6 — Answer generation

- Implement the configurable LLM client.
- Implement `/ask`.
- Build a grounded RAG prompt.
- Deduplicate final child hits by parent ID, batch-fetch matching parents from PostgreSQL, preserve retrieval rank, enforce the generation-context budget, and handle missing or mismatched parents with child-text fallback.
- Include citations in responses.
- Handle insufficient context, PostgreSQL lookup failures, and LLM API failures explicitly.

**Acceptance:** The assistant produces source-backed answers using parent context resolved from child matches, with traceable citations and bounded context.

### Phase 7 — Evaluation and documentation

- Create a 20-question evaluation dataset.
- Implement retrieval metrics.
- Measure retrieval and ingestion timings.
- Compare dense-only, BM25-only, fused, and reranked relevance on the same questions, including code/keyword cases and semantic paraphrases.
- Test grounded answer generation.
- Document architecture, installation, endpoints, and limitations.

**Acceptance:** The system can be evaluated reproducibly and run locally following the README.

## 13. Testing and Code Quality

- Use Python type hints.
- Use Pydantic v2 for validation.
- Configure Ruff.
- Use pytest for unit and integration tests.
- Mock external HTTP and LLM calls in unit tests.
- Test failed downloads and incomplete files.
- Test short and oversized Markdown sections.
- Test per-document chunker parent resolution, non-null parent IDs, matching parent/child revisions, and deterministic chunking from unchanged local sources.
- Test PostgreSQL parent round trips, preserved Markdown rendering metadata, batched ID lookups, schema migrations, and ingestion-state persistence.
- Verify parent lookup after restarting the application without local source reconstruction or loading the complete parent corpus.
- Test repeated headings, code-fence heading lookalikes, source ranges, and complete embedding-input token limits.
- Test hashing and deterministic identifiers.
- Test repeated ingestion and document updates.
- Test chunking-configuration changes, failed parent transactions, interrupted child upserts, retries, cleanup failures, obsolete-child removal before obsolete-parent deletion, and completion only after both stores are consistent.
- Test Qdrant retrieval using an isolated test collection.
- Test adding sparse schema to the pinned real Qdrant server without recreating the dense collection, preserving dense vectors/IDs/payloads during sparse-only updates, complete sparse coverage, restart behavior, interrupted backfill recovery, and sparse configuration invalidation.
- Test document/query lexical preprocessing parity, stable term indices, identifier aliases and exact payload matching, case/punctuation/digit handling, long tokens, nonempty child vectors, invalid indices/weights, and empty lexical questions.
- Test filtered dense/sparse prefetch limits, RRF deduplication and fused limits, sparse-only relevant hits, unavailable branch diagnostics, and explicit branch-error handling.
- Use an isolated PostgreSQL test database/schema for integration tests; keep unit tests independent of running databases.
- Test Top-3 child semantics, one batched PostgreSQL parent fetch, preserved ranking, parent deduplication, missing/mismatched-parent fallback, explicit PostgreSQL outage handling, context budgeting, and citation provenance.
- Test candidate-pool limits, reranker ordering and tie-breaking, reranker input limits, model-load reuse, and explicit degraded behavior when reranking fails.
- Add structured logs for ingestion and retrieval operations.
- Never commit API keys or other secrets.

## 14. Out of Scope

Do not implement:

- Multiple Qdrant collections for experiments
- Fixed-size chunking strategy
- Chunk-size comparison experiments
- Overlap comparison experiments
- Top-K comparison experiments
- LangGraph
- AI agents
- Query rewriting and learned sparse models such as SPLADE
- A separate lexical search service or full-text BM25 engine
- Frontend
- Conversation memory
- Authentication
- Kubernetes
- Cloud deployment
- Background ingestion jobs

## 15. Definition of Done

The project is complete when:

1. Ten OWASP Markdown files can be downloaded directly from GitHub.
2. Files are saved under `data/owasp/`.
3. Combined Markdown-aware parent–child chunking works with the selected configuration and produces bounded per-document parents and children.
4. Local dense embeddings and BM25 sparse vectors are generated successfully for every searchable child, including existing-point backfill.
5. PostgreSQL durably stores parents and ingestion state; one Qdrant collection stores both child vector representations with versioned sparse configuration and IDF enabled. Parents can be fetched by ID after restart without rebuilding them from local sources or retaining the corpus in memory.
6. Every child point contains a valid parent ID, document ID, title, filename, source metadata, and hashes.
7. Ingestion is idempotent across PostgreSQL and Qdrant, with safe cleanup ordering and recovery from partial failures.
8. `/search` prefetches up to 20 dense and 20 BM25 child hits, fuses to up to 20 unique candidates with RRF, reranks them with the cross-encoder, and returns the relevant Top-3 children with correctly labeled scores, retrieval mode, and parent IDs. Code/keyword and semantic retrieval are validated separately.
9. `/ask` deduplicates parent IDs, batch-fetches matching parents from PostgreSQL in retrieval rank order, and generates grounded answers with traceable citations within the generation model's context budget. Missing or mismatched parents use child-text fallback; database outages are reported explicitly.
10. Retrieval evaluation and tests run successfully.
11. The README contains installation, usage, architecture, and evaluation instructions.

## 16. Instructions for Codex

Implement this project incrementally, starting with Phase 1.

Before writing code, inspect the existing project structure and prepare a concise implementation checklist.

Follow the specified architecture and configuration. Do not introduce extra chunking strategies, collections, evaluation experiments, or agent frameworks.

After each phase:

- Run relevant tests.
- Run Ruff.
- Verify acceptance criteria.
- Summarize changes and technical decisions.
- Identify unresolved issues.

Use simple, readable, maintainable Python code. Prefer established libraries and explicit behavior over excessive abstraction.

The priority is to understand and implement the complete RAG pipeline while keeping the project small and suitable for local development.
