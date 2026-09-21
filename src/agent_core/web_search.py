from __future__ import annotations

import re
from html import unescape
from typing import Any, Dict, List
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


def _clean_text(value: str) -> str:
    value = unescape(value or "")
    value = re.sub(r"<.*?>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_search_results(html: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Parse common search engine HTML into a list of {title, link, snippet} dictionaries."""
    results: List[Dict[str, str]] = []
    try:
        links = re.findall(r'<a[^>]+class=["\'][^"\']*(?:result-link|result-link--title|result__a)[^"\']*["\'][^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, flags=re.I | re.S)
        for href, title_html in links[: max_results * 3]:
            href = href.strip()
            if not href or href.startswith("javascript:"):
                continue
            title = _clean_text(title_html)
            if not title:
                continue
            # Find the nearest snippet after the link in a common search layout.
            snippet = ""
            match = re.search(rf'<a[^>]+href=["\']{re.escape(href)}["\'][^>]*>.*?</a>(?:\s*<div[^>]*class=["\'][^"\']*(?:result-snippet|snippet|result__snippet)[^"\']*["\'][^>]*>)(.*?)</div>', html, flags=re.I | re.S)
            if match:
                snippet = _clean_text(match.group(1))
            elif re.search(rf'<a[^>]+href=["\']{re.escape(href)}["\'][^>]*>.*?</a>(?:\s*<span[^>]*class=["\'][^"\']*(?:result-snippet|snippet)[^"\']*["\'][^>]*>)(.*?)</span>', html, flags=re.I | re.S):
                match = re.search(rf'<a[^>]+href=["\']{re.escape(href)}["\'][^>]*>.*?</a>(?:\s*<span[^>]*class=["\'][^"\']*(?:result-snippet|snippet)[^"\']*["\'][^>]*>)(.*?)</span>', html, flags=re.I | re.S)
                snippet = _clean_text(match.group(1)) if match else ""
            if snippet:
                results.append({"title": title, "link": href, "snippet": snippet})
            else:
                results.append({"title": title, "link": href, "snippet": ""})
            if len(results) >= max_results:
                break
    except Exception:
        return []

    if not results:
        # Generic fallback: capture the first available link + adjacent text
        generic_links = re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, flags=re.I | re.S)
        for href, title_html in generic_links[:max_results]:
            title = _clean_text(title_html)
            if title:
                results.append({"title": title, "link": href, "snippet": ""})
    return results[:max_results]


def search_web(query: str, max_results: int = 5, timeout: int = 12) -> List[Dict[str, str]]:
    """Search the web using a lightweight, dependency-free DuckDuckGo HTML endpoint."""
    query = (query or "").strip()
    if not query:
        return []

    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urlopen(req, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="ignore")
            return extract_search_results(html, max_results=max_results)
    except Exception:
        return []


def format_search_summary(query: str, results: List[Dict[str, str]], max_items: int = 3) -> str:
    """Build a concise, executive-style summary from search results."""
    if not results:
        return ""
    lines = [f"External benchmark scan for '{query}':"]
    for idx, item in enumerate(results[:max_items], start=1):
        title = item.get("title", "Result")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        lines.append(f"{idx}. {title}")
        if link:
            lines.append(f"   Link: {link}")
        if snippet:
            lines.append(f"   Insight: {snippet[:180]}")
    return "\n".join(lines)
