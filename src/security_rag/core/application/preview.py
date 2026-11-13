"""Inspect chunk contents and token usage without indexing."""

from collections.abc import Iterable

from security_rag.core.application.ports import Chunker
from security_rag.core.domain.documents import Document


def preview_chunks(documents: Iterable[Document], chunker: Chunker) -> dict:
    reports = []
    for document in documents:
        chunks = chunker.chunk(document)
        children_by_parent = {}
        for child in chunks.children:
            children_by_parent.setdefault(child.parent_id, []).append(child)
        parent_chunks = [
            {
                "id": str(parent.id),
                "text": parent.text,
                "children": [
                    {"id": str(child.id), "text": child.text}
                    for child in children_by_parent[parent.id]
                ],
            }
            for parent in chunks.parents
        ]
        reports.append(
            {
                "document_id": document.metadata.document_id,
                "document_hash": document.metadata.document_hash,
                "chunking_config_hash": chunks.parents[0].chunking_config_hash,
                "parents": len(chunks.parents),
                "children": len(chunks.children),
                "max_parent_tokens": max(chunker.parent_token_count(p) for p in chunks.parents),
                "max_child_tokens": max(chunker.child_token_count(c) for c in chunks.children),
                "parent_chunks": parent_chunks,
            }
        )
    return {
        "parents": sum(report["parents"] for report in reports),
        "children": sum(report["children"] for report in reports),
        "max_parent_tokens": max(report["max_parent_tokens"] for report in reports),
        "max_child_tokens": max(report["max_child_tokens"] for report in reports),
        "documents": reports,
    }
