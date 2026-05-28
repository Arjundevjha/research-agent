"""
Free research tools — no paid API keys required.

- DuckDuckGo search: web search with zero API keys
- Page scraper: extracts text content from URLs using requests + BeautifulSoup
"""

import requests
from crewai.tools import tool
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup


@tool("Web Search")
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo. Returns top results with titles,
    snippets, and URLs. Completely free — no API key needed.

    Args:
        query: The search query string.
    """
    try:
        results = DDGS().text(query, max_results=8)
        if not results:
            return f"No results found for: {query}"

        output_parts = []
        for i, r in enumerate(results, 1):
            output_parts.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('href', 'No URL')}\n"
                f"    Snippet: {r.get('body', 'No snippet')}\n"
            )
        return "\n".join(output_parts)
    except Exception as e:
        return f"Search failed: {e}"


@tool("Scrape Webpage")
def scrape_webpage(url: str) -> str:
    """
    Scrape the text content from a webpage. Extracts the title, any visible
    author/date metadata, and the main body text.

    Args:
        url: The full URL of the page to scrape.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script/style/nav elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract metadata
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Try to find author
        author = ""
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta:
            author = author_meta.get("content", "")

        # Try to find publication date
        date = ""
        for attr_name in ["article:published_time", "date", "publishdate"]:
            date_meta = soup.find("meta", attrs={"property": attr_name}) or soup.find(
                "meta", attrs={"name": attr_name}
            )
            if date_meta:
                date = date_meta.get("content", "")
                break

        # Extract main text
        body_text = soup.get_text(separator="\n", strip=True)

        # Truncate to avoid token overflow — keep first ~6000 chars
        max_chars = 6000
        if len(body_text) > max_chars:
            body_text = body_text[:max_chars] + "\n\n[... content truncated ...]"

        metadata = f"Title: {title}\nAuthor: {author}\nDate: {date}\nURL: {url}\n"
        return f"--- Page Metadata ---\n{metadata}\n--- Page Content ---\n{body_text}"

    except requests.exceptions.Timeout:
        return f"Timeout: Page at {url} took too long to respond."
    except requests.exceptions.RequestException as e:
        return f"Failed to scrape {url}: {e}"
