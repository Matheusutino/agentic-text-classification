from __future__ import annotations

from pydantic import BaseModel


class ArxivArticle(BaseModel):
    arxiv_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    pdf_url: str | None = None
    entry_url: str | None = None


class ArxivSearchResult(BaseModel):
    query: str
    start: int
    max_results: int
    total_results: int
    articles: list[ArxivArticle]


class DDGSearchResultItem(BaseModel):
    title: str | None = None
    link: str | None = None
    snippet: str | None = None
    source: str | None = None
    date: str | None = None


class DDGSearchResult(BaseModel):
    query: str
    backend: str
    max_results: int
    results: list[DDGSearchResultItem]


class URLContentResult(BaseModel):
    url: str
    final_url: str | None = None
    title: str | None = None
    markdown: str
