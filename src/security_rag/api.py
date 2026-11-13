"""FastAPI entry point for child retrieval."""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from security_rag.bootstrap import search_runtime
from security_rag.config import Settings
from security_rag.core.application.ports import SearchFilters
from security_rag.core.application.search import InvalidSearchQuery, SearchResult
from security_rag.core.domain.lexical import IDENTIFIER, identifiers

logger = logging.getLogger(__name__)
QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
FilterText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: QueryText
    document_id: FilterText | None = None
    identifier: FilterText | None = None

    @field_validator("identifier")
    @classmethod
    def canonical_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if IDENTIFIER.fullmatch(value) is None:
            raise ValueError("Expected one supported exact identifier")
        return identifiers(value)[0]


def create_app(
    search: Callable[[str, SearchFilters], SearchResult] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if search is not None:
            app.state.search = search
            yield
        else:
            with search_runtime(Settings()) as run:
                app.state.search = run
                yield

    app = FastAPI(title="AI Security Knowledge Assistant", lifespan=lifespan)

    @app.post("/search", response_model=SearchResult)
    def search_endpoint(body: SearchRequest, request: Request) -> SearchResult:
        try:
            return request.app.state.search(
                body.query, SearchFilters(body.document_id, body.identifier)
            )
        except InvalidSearchQuery as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.warning("search_unavailable error_type=%s", type(exc).__name__)
            raise HTTPException(
                status_code=503, detail="Search is temporarily unavailable"
            ) from exc

    return app


app = create_app()
