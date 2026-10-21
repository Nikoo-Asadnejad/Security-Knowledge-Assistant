# AI Security Knowledge Assistant — Implementation Plan

## 1. Project Overview

Build a RAG-powered AI Security Knowledge Assistant using the official OWASP Top 10 for LLM Applications (2026).

The application answers questions about LLM vulnerabilities, security risks, attack scenarios, and mitigations using retrieved OWASP documentation.

**Primary goal:** Learn RAG engineering through practical implementation of document ingestion, Markdown-aware chunking, embeddings, Qdrant, semantic retrieval, grounded generation, and evaluation.

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
| LLM | Configurable OpenAI-compatible API |
| Chunking | LangChain text splitters |
| Tokenizer | Embedding-model-compatible tokenizer |
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

Implement **only Markdown-aware hierarchical chunking**.

Do not implement fixed-size chunking as a separate strategy.

### Configuration

| Parameter | Value |
|---|---|
| Strategy | Markdown-aware |
| Chunk size | 450 tokens |
| Overlap between sections | 0 |
| Overlap within oversized sections | 60 tokens |
| Embedding model | BAAI/bge-small-en-v1.5 |

### Chunking algorithm

**Step 1 — Parse Markdown**

Identify document headings and their hierarchy:

- `#` — Document title
- `##` — Section
- `###` — Subsection
- `####` — Nested subsection

Preserve the document title, heading path, section metadata, and original Markdown structure.

**Step 2 — Split by sections**

Split content using Markdown heading boundaries.

Each section becomes an independent logical unit.

Do not overlap content between unrelated sections.

**Step 3 — Handle chunk sizes**

For each section:

- If the section fits within 450 tokens, keep it as one chunk.
- If the section exceeds 450 tokens, split it recursively.
- Use a 60-token overlap only between chunks originating from the same oversized section.
- Prefer paragraph and sentence boundaries.
- Preserve Markdown lists and code fences where possible.
- Avoid empty chunks and heading-only chunks.

Use the embedding model's tokenizer to validate chunk lengths.

The embedding model supports up to 512 input tokens. Account for special tokens and any title or heading information added to the embedded text so input is not silently truncated.

**Step 4 — Add semantic context**

Include the document title and heading path in the text submitted for embedding.

For example:

```text
Document: Prompt Injection
Section: Prevention and Mitigation Strategies

[Actual section content...]
```

Keep the original chunk content separately in the payload.

**Step 5 — Hash chunks**

Calculate:

- `document_hash`: SHA-256 of the complete source file.
- `chunk_hash`: SHA-256 of the final chunk text.

## 5. Qdrant Vector Storage

Use exactly one Qdrant collection:

`owasp_security_2026`

Embedding model:

`BAAI/bge-small-en-v1.5`

Vector dimensions: 384.

### Qdrant point payload

```json
{
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
  "source_path": "data/owasp/LLM01_PromptInjection.md",
  "source_url": "https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM01_PromptInjection.md",
  "document_hash": "<sha256>",
  "chunk_hash": "<sha256>",
  "version": "2026"
}
```

### Point identification

Generate deterministic UUIDs using:

- Document ID
- Chunk position
- Chunking configuration

Do not use random IDs for repeated ingestion.

### Idempotent ingestion

- Detect whether the document has already been indexed.
- Compare stored `document_hash` with the current file hash.
- Skip unchanged documents.
- Reprocess changed documents.
- Remove obsolete points when documents are updated.
- Avoid duplicate vectors and stale chunks.
- Handle indexing failures without incorrectly marking a document as successfully indexed.

Keep the Qdrant collection name configurable, but use only one collection for this project.

## 6. Embedding Pipeline

Use FastEmbed to generate embeddings locally on the CPU.

The ingestion pipeline:

1. Read locally downloaded Markdown files.
2. Parse Markdown headings.
3. Generate chunks.
4. Calculate hashes and metadata.
5. Prepare embedding text with contextual headings.
6. Generate embeddings in batches.
7. Upsert points into Qdrant.
8. Log ingestion statistics.

Track:

- Number of documents processed
- Number of chunks created
- Embedding duration
- Indexing duration
- Documents skipped as unchanged
- Failed documents

Embedding should be performed offline through the ingestion CLI, not during FastAPI startup.

## 7. Semantic Retrieval

Implement semantic similarity search using Qdrant.

Retrieval configuration:

**Top-K = 3**

### Retrieval flow

1. Receive user question.
2. Generate query embedding using the same model.
3. Search the single Qdrant collection.
4. Retrieve the three most similar chunks.
5. Return content, scores, titles, sections, filenames, and source URLs.

Support optional filtering by `document_id`.

Do not implement hybrid search, reranking, or query rewriting in the MVP.

## 8. RAG Answer Generation

Use a straightforward two-step RAG pipeline:

```text
User Question
    ↓
Query Embedding
    ↓
Qdrant Similarity Search
    ↓
Top-3 Retrieved Chunks
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

Return the Top-3 retrieved chunks.

Each result includes:

- Text
- Similarity score
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

- Markdown-aware chunking
- 450-token maximum
- 60-token overlap within oversized sections
- No overlap between sections
- Top-3 retrieval

Focus evaluation on the quality of this configuration.

### Evaluation dataset

Create approximately 20 manually reviewable test questions covering:

- Vulnerability definitions
- Attack scenarios
- Mitigation strategies
- Differences between security vulnerabilities
- Questions requiring evidence from multiple sections

For every question, identify expected relevant documents and sections.

### Evaluation metrics

Measure:

- Recall@3
- MRR@3
- Retrieval latency
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

### Phase 2 — Markdown chunking

- Implement heading-aware parsing.
- Configure 450-token chunks.
- Implement conditional 60-token overlap for oversized sections.
- Add document title and heading context to embedding text.
- Implement chunk hashing.
- Validate chunk sizes using the embedding tokenizer.
- Test section boundaries, short sections, long sections, lists, and code fences.

**Acceptance:** Every chunk preserves useful semantic context, respects the token budget, and has valid metadata.

### Phase 3 — Embeddings and Qdrant

- Initialize FastEmbed.
- Generate embeddings locally.
- Create the Qdrant collection.
- Store points and metadata.
- Implement deterministic point IDs.
- Implement incremental, idempotent ingestion.
- Verify stored payloads.

**Acceptance:** All ten OWASP documents are indexed without duplicates, and stored points contain titles, filenames, hashes, and source links.

### Phase 4 — Search API

- Implement semantic retrieval.
- Implement `/search`.
- Use Top-3 retrieval.
- Add optional document filtering.
- Return source metadata and similarity scores.
- Add retrieval tests.

**Acceptance:** Representative questions retrieve relevant OWASP content.

### Phase 5 — Answer generation

- Implement the configurable LLM client.
- Implement `/ask`.
- Build a grounded RAG prompt.
- Include citations in responses.
- Handle insufficient context and API failures.

**Acceptance:** The assistant produces source-backed answers using retrieved chunks.

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
- Test hashing and deterministic identifiers.
- Test repeated ingestion and document updates.
- Test Qdrant retrieval using an isolated test collection.
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
- Reranking
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
3. Markdown-aware chunking works with the selected configuration.
4. Local embeddings are generated successfully.
5. One Qdrant collection stores all document chunks.
6. Every point contains document ID, title, filename, source metadata, and hashes.
7. Ingestion is idempotent.
8. `/search` returns relevant Top-3 chunks.
9. `/ask` generates grounded answers with citations.
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