# AI Security Knowledge Assistant — Implementation Plan

## 1. Project Overview

Build a RAG-powered AI Security Knowledge Assistant using the official OWASP Top 10 for LLM Applications (2026).

The application answers questions about LLM vulnerabilities, security risks, attack scenarios, and mitigations using retrieved OWASP documentation.

**Primary goal:** Learn RAG engineering through practical implementation of document ingestion, Markdown-aware parent–child chunking, embeddings, Qdrant, semantic retrieval, grounded generation, and evaluation.

Keep the project small, modular, maintainable, and suitable for running locally.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Package manager | uv |
| API | FastAPI |
| Validation | Pydantic v2 |
| Vector database | Qdrant (Docker) |
| Embeddings | FastEmbed — BAAI/bge-small-en-v1.5 |
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

Every searchable child chunk must contain a non-null `parent_id` that resolves to an in-memory parent record. A heading path alone does not provide this relationship.

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

Each parent record has an `id`, `document_id`, `section`, `heading_path`, `text`, `parent_index`, `source_start`, `source_end`, `document_hash`, `parent_hash`, and `chunking_config_hash`, plus the document's title, filename, version, and source metadata. Source ranges are zero-based, half-open character offsets into the decoded original document; `parent_hash` hashes the parent text. Parents are in-memory context records and are not embedded.

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

Keep parents in memory as part of the chunker's output. Do not save parent records to files or a database. The `chunk` CLI validates parents and children and prints statistics without writing chunk files.

Keep parent text in one place rather than duplicating it in every child payload. In the answer-generation phase, rebuild parents from the local Markdown documents using the same tokenizer and chunking configuration when needed after restart. Only resolve a child to a parent with matching IDs, document hash, and configuration hash. If matching source or configuration is unavailable, use the child-text fallback in Section 7. No additional storage service or Qdrant collection is required.

### Design references

- [LangChain Markdown splitting](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter): heading metadata, per-section recursive splitting, overlap behavior, and formatting limitations.
- [LangChain parent-document retrieval implementation](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/retrievers/parent_document_retriever.py): searching small children and resolving their stored parent IDs. This is a design reference, not a required dependency.
- [BAAI model card](https://huggingface.co/BAAI/bge-small-en-v1.5): 384-dimensional embeddings and a 512-token input limit.
- [Cross-encoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2): reference implementation and passage-ranking model details.

## 5. Qdrant Vector Storage

Use exactly one Qdrant collection:

`owasp_security_2026`

Embedding model:

`BAAI/bge-small-en-v1.5`

Vector dimensions: 384.

Store only searchable children as vector points. Parent records remain in memory and can be rebuilt from matching local sources as described in Section 4.

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

### Point identification

Use the deterministic parent and child UUID rules from Section 4. The Qdrant point ID must equal the payload's child `id`. `chunk_index` is document-wide; `child_index` is relative to the parent. Do not use random IDs for repeated ingestion.

### Idempotent ingestion

- Detect whether the document has already been indexed.
- Compare both `document_hash` and `chunking_config_hash` with the completed ingestion state.
- Skip documents only when source, configuration, and stored child records are complete and unchanged, and matching parents can be rebuilt from local sources.
- Reprocess changed documents or chunking configurations, including migration from chunks without `parent_id`.
- Validate child references against the new revision's in-memory parents before upserting child points.
- After all new child upserts succeed, remove obsolete child points, and only then mark ingestion complete.
- Avoid duplicate vectors and stale chunks.
- Handle failures without incorrectly marking a document as successfully indexed; retries must converge using deterministic IDs. Never resolve children to parents from a different source/configuration revision.

For the local MVP, run ingestion while the API is stopped. Concurrent reindexing and parent-cache refresh are out of scope.

Keep the Qdrant collection name configurable, but use only one collection for this project.

## 6. Embedding Pipeline

Use FastEmbed to generate embeddings locally on the CPU.

The ingestion pipeline:

1. Read locally downloaded Markdown files.
2. Parse Markdown headings.
3. Generate bounded Markdown parents and their searchable children.
4. Calculate hashes, deterministic IDs, parent links, and source metadata.
5. Prepare and validate child embedding inputs with contextual headings.
6. Generate child embeddings in batches; do not embed parents.
7. Validate in-memory parent references, upsert child points into Qdrant, and complete revision cleanup as specified in Section 5.
8. Log ingestion statistics.

The cross-encoder is loaded and used at retrieval time, not during ingestion.

Track:

- Number of documents processed
- Number of parents and children created
- Child token lengths and invalid parent-reference failures
- Embedding duration
- Indexing duration
- Documents skipped as unchanged
- Failed documents

Embedding should be performed offline through the ingestion CLI, not during FastAPI startup.

## 7. Semantic Retrieval

Implement a two-stage retrieval pipeline: dense semantic candidate retrieval using Qdrant, followed by cross-encoder reranking.

Retrieval configuration:

**Candidate-K = 20; final-K = 3**

### Retrieval flow

1. Receive user question.
2. Generate query embedding using the same model.
3. Search the single Qdrant collection.
4. Retrieve up to 20 highest-scoring child candidates, or fewer when the collection contains fewer points.
5. Score each `(question, child embedding text)` pair with the configured cross-encoder. Run inference in batches, with truncation disabled or explicitly validated against the reranker model's maximum input length.
6. Sort by cross-encoder score descending, use the original dense score only as a deterministic tie-breaker, and retain the final three children, or fewer when fewer are available.
7. Return child IDs, parent IDs, dense scores, reranker scores, titles, sections, filenames, and source URLs.

The cross-encoder score is the final retrieval rank signal; it is not comparable to the Qdrant similarity score and must not be blended unless a later evaluated change specifies normalization and weights. A cross-encoder is query-dependent and cannot be precomputed during ingestion, so load it once per process and reuse it across requests. If reranking fails, log the failure and return the dense candidates only when the API contract explicitly marks the response as degraded; do not silently present dense scores as reranker scores.

For `/ask`, resolve the returned `parent_id` values against in-memory parents rebuilt from matching local sources and use the parents as generation context:

- Deduplicate parents while preserving the rank of their highest-scoring matching child.
- Keep matched child IDs and scores for traceability. Scores belong to child matches, not separately scored parents.
- Expand only to the immediate parent; do not recursively fetch the whole document or unrelated siblings.
- Top-3 means the three reranked child hits, which can yield one to three distinct parents. Do not fetch extra hits merely to fill three parents.
- If a parent is missing or its document/configuration revision differs from the child, log an integrity error and fall back to that child's text. Never substitute content from a different revision.
- Enforce the generation model's context budget after reserving space for instructions, the question, formatting, and output. Prefer full parents; when a parent does not fit, use its matching children if they fit. Omit remaining units in rank order rather than silently truncating them, and report insufficient evidence when necessary.

Support optional filtering by `document_id` before reranking.

Do not implement hybrid search or query rewriting in the MVP.

## 8. RAG Answer Generation

Use a straightforward retrieval-and-generation RAG pipeline:

```text
User Question
    ↓
Query Embedding
    ↓
Qdrant Similarity Search
    ↓
Cross-Encoder Reranking (top 20 candidates)
    ↓
Top-3 Reranked Children
    ↓
Resolve Parent IDs and Deduplicate Parents
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

Return service health and Qdrant connection status.

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
- Dense similarity score
- Cross-encoder reranker score
- Document ID
- Title
- File name
- Section
- Source URL

Optional document filtering can be supported.

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
- Ingesting documents into Qdrant
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
- Retrieve 20 dense child candidates, rerank with the cross-encoder, keep the top 3 children, then expand to parents for answers

Focus evaluation on the quality of this configuration.

### Evaluation dataset

Create approximately 20 manually reviewable test questions covering:

- Vulnerability definitions
- Attack scenarios
- Mitigation strategies
- Differences between security vulnerabilities
- Questions requiring evidence from multiple sections

For every question, identify expected relevant documents and sections. Distinguish evidence directly present in retrieved children from evidence added by parent expansion.

### Evaluation metrics

Measure:

- Dense-candidate Recall@20
- Reranked Recall@3 and MRR@3 over final child hits, using expected document/section relevance labels
- Relevant-section coverage in the final expanded context after budgeting, reported separately from child retrieval metrics
- Number of distinct parents included and generation-context token count
- Dense retrieval latency, reranking latency, and end-to-end retrieval latency
- Embedding duration
- Indexing duration

Record results as JSON or CSV.

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
│       │   └── embedder.py
│       ├── vector_store/
│       │   └── qdrant_store.py
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
- Keep parent records in memory without saving chunk files.
- Validate chunk sizes using the embedding tokenizer.
- Test relative heading levels, repeated headings, introductory text, short/long sections, oversized parents, lists, tables, and code fences.
- Test that every child resolves to the correct parent and that overlap never crosses parent or heading boundaries.

**Acceptance:** Every child preserves useful semantic context, respects the complete-input token budget, and resolves to one in-memory parent from the same document/configuration revision. Parents preserve source content within the parent token limit. The chunk command writes no parent or child files.

### Phase 3 — Embeddings and Qdrant

- Initialize FastEmbed.
- Generate child embeddings locally.
- Create the Qdrant collection.
- Store points and metadata.
- Implement deterministic point IDs.
- Implement incremental, idempotent ingestion with parent-reference validation and obsolete-child cleanup.
- Verify stored payloads.

**Acceptance:** All ten OWASP documents are indexed without duplicates or dangling parent references; stored child points contain parent IDs, titles, filenames, hashes, and source links, and unchanged complete revisions can be skipped safely.

### Phase 4 — Search API

- Implement semantic retrieval.
- Retrieve a 20-child dense candidate pool and rerank it with the cross-encoder in batches.
- Implement `/search`.
- Return the top three reranked children with both dense and reranker scores.
- Add optional document filtering.
- Return child IDs, parent IDs, source metadata, and child similarity scores.
- Add retrieval tests.

**Acceptance:** Representative questions retrieve relevant OWASP content, and reranking is observable through scores and improves or preserves the labeled top-3 retrieval results.

### Phase 5 — Answer generation

- Implement the configurable LLM client.
- Implement `/ask`.
- Build a grounded RAG prompt.
- Rebuild parents from matching local sources, resolve and deduplicate them, enforce the generation-context budget, and handle missing parents with child-text fallback.
- Include citations in responses.
- Handle insufficient context and API failures.

**Acceptance:** The assistant produces source-backed answers using parent context resolved from child matches, with traceable citations and bounded context.

### Phase 6 — Evaluation and documentation

- Create a 20-question evaluation dataset.
- Implement retrieval metrics.
- Measure retrieval and ingestion timings.
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
- Test in-memory parent resolution, non-null parent IDs, matching parent/child revisions, and deterministic reconstruction from unchanged local sources.
- Test repeated headings, code-fence heading lookalikes, source ranges, and complete embedding-input token limits.
- Test hashing and deterministic identifiers.
- Test repeated ingestion and document updates.
- Test chunking-configuration changes, interrupted child upserts, retries, and obsolete-child cleanup.
- Test Qdrant retrieval using an isolated test collection.
- Test Top-3 child semantics, parent deduplication, missing-parent fallback, context budgeting, and citation provenance.
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
- Hybrid search
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
3. Combined Markdown-aware parent–child chunking works with the selected configuration and produces bounded in-memory parents.
4. Local embeddings are generated successfully.
5. One Qdrant collection stores all child vectors; matching parents can be rebuilt from local sources after restart, with child-text fallback when unavailable.
6. Every child point contains a valid parent ID, document ID, title, filename, source metadata, and hashes.
7. Ingestion is idempotent.
8. `/search` retrieves up to 20 dense candidates, reranks them with the cross-encoder, and returns the relevant Top-3 child chunks with both scores and parent IDs.
9. `/ask` resolves and deduplicates parents and generates grounded answers with traceable citations within the generation model's context budget.
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
