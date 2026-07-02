from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from src.types import (
    ArxivArticle,
    ArxivSearchResult,
    DDGSearchResult,
    DDGSearchResultItem,
    URLContentResult,
)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _find_text(element: ET.Element, path: str) -> str:
    child = element.find(path, ATOM_NS)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def search_arxiv(
    query: str,
    max_results: int = 5,
    start: int = 0,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> ArxivSearchResult:
    """Search scholarly articles in arXiv.

    Args:
        query: arXiv API query string, such as `all:transformer` or `cat:cs.CL`.
        max_results: Number of records requested. Keep this small for interactive usage.
        start: Zero-based offset into the arXiv result set.
        sort_by: Sort field accepted by arXiv (`relevance`, `lastUpdatedDate`, `submittedDate`).
        sort_order: Sort order accepted by arXiv (`ascending`, `descending`).

    Example:
        `search_arxiv("all:transformer", max_results=5, start=0, sort_by="relevance", sort_order="descending")`

    Returns:
        A structured list of arXiv articles and metadata about the search result set.

    Raises:
        ValueError: If the requested paging arguments are outside arXiv practical limits.
    """
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")
    if max_results > 2000:
        raise ValueError("max_results must be at most 2000 per arXiv request slice.")
    if start < 0:
        raise ValueError("start must be non-negative.")
    if start + max_results > 30000:
        raise ValueError("arXiv requests must stay within the first 30000 results.")

    params = urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
    )
    url = f"{ARXIV_API_URL}?{params}"
    with urlopen(url, timeout=30) as response:
        payload = response.read()

    root = ET.fromstring(payload)
    total_results_text = _find_text(root, "{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    total_results = int(total_results_text) if total_results_text else 0

    articles: list[ArxivArticle] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        authors = [
            _find_text(author, "atom:name")
            for author in entry.findall("atom:author", ATOM_NS)
            if _find_text(author, "atom:name")
        ]
        categories = [
            category.attrib.get("term", "").strip()
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term")
        ]

        pdf_url = None
        entry_url = _find_text(entry, "atom:id") or None
        for link in entry.findall("atom:link", ATOM_NS):
            title = link.attrib.get("title")
            href = link.attrib.get("href")
            if title == "pdf" and href:
                pdf_url = href
                break

        article_id = entry_url.rsplit("/", 1)[-1] if entry_url else ""
        articles.append(
            ArxivArticle(
                arxiv_id=article_id,
                title=_find_text(entry, "atom:title"),
                summary=_find_text(entry, "atom:summary"),
                authors=authors,
                categories=categories,
                published=_find_text(entry, "atom:published"),
                updated=_find_text(entry, "atom:updated"),
                pdf_url=pdf_url,
                entry_url=entry_url,
            )
        )

    return ArxivSearchResult(
        query=query,
        start=start,
        max_results=max_results,
        total_results=total_results,
        articles=articles,
    )


def search_ddg(
    query: str,
    max_results: int = 5,
    backend: str = "text",
) -> DDGSearchResult:
    """Search the public internet with DuckDuckGo through LangChain's official DDG integration.

    Args:
        query: Internet search query string.
        max_results: Maximum number of web results to return.
        backend: Search backend. Common values are `text` for general web search and `news` for news search.

    Example:
        `search_ddg("best tf-idf settings for text classification", max_results=5, backend="text")`

    Returns:
        Structured DuckDuckGo search results from the public web.

    Use this when a general internet search is genuinely useful, such as:
        - finding recent discussions, tutorials, or documentation;
        - checking current terminology, libraries, or common practices;
        - gathering web evidence to justify a modeling choice.
    """
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")

    try:
        from langchain_community.tools import DuckDuckGoSearchResults
    except Exception as exc:
        raise ImportError(
            "DuckDuckGo search requires `langchain-community` and `ddgs`."
        ) from exc

    tool = DuckDuckGoSearchResults(
        output_format="list",
        max_results=max_results,
        backend=backend,
    )
    raw_results = tool.invoke(query)
    results: list[DDGSearchResultItem] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append(
            DDGSearchResultItem(
                title=item.get("title"),
                link=item.get("link") or item.get("url"),
                snippet=item.get("snippet") or item.get("body"),
                source=item.get("source"),
                date=item.get("date"),
            )
        )

    return DDGSearchResult(
        query=query,
        backend=backend,
        max_results=max_results,
        results=results,
    )


async def _fetch_url_content_async(url: str) -> URLContentResult:
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url)

    return URLContentResult(
        url=url,
        final_url=getattr(result, "url", None),
        title=getattr(result, "title", None),
        markdown=getattr(result, "markdown", "") or "",
    )


def fetch_url_content(url: str) -> URLContentResult:
    """Fetch the main textual content of a web page from a URL.

    Use this after a search tool returns relevant links and you want the page content itself.
    The result is extracted as markdown so the agent can inspect the article or page text.

    Args:
        url: Public URL to crawl and extract content from.

    Example:
        `fetch_url_content("https://example.com/article")`

    Returns:
        The crawled page content as markdown, plus URL metadata.
    """
    return asyncio.run(_fetch_url_content_async(url))
