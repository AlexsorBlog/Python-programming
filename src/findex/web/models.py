"""Pydantic lives only at the edge. Inside, the dataclasses from labs 2-3 stay."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ScorerName(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


class Hit(BaseModel):
    doc_id: int
    score: float
    title: str
    snippet: str = ""


class SearchResponse(BaseModel):
    query: str
    scorer: ScorerName
    page: int
    k: int
    total: int
    pages: int
    hits: list[Hit]
    took_ms: float


class DocResponse(BaseModel):
    doc_id: int
    title: str
    length: int
    text: str


class StatsResponse(BaseModel):
    terms: int
    documents: int
    avg_doc_length: float
    positions: bool
    repr: str


class HealthResponse(BaseModel):
    status: str
    documents: int = Field(default=0)
