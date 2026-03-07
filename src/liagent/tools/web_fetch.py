"""Web page fetch tool — Markdown-for-Agents with Playwright fallback."""

import asyncio
import atexit
import re

import httpx

from . import ToolCapability, tool
from ._utils import _sanitize

_TIMEOUT_MS = 15_000
_WAIT_STRATEGY = "commit"  # fastest: returns on first byte, JS settles during _JS_SETTLE_MS
_JS_SETTLE_MS = 2000  # wait for JS to render after initial response
_MAX_OUTPUT_CHARS = 3000
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# JS injected before every page to mask headless fingerprints.
_STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => false});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en', 'zh-CN']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = {runtime: {}};
"""

# Persistent browser — lazy-init, reused across calls, cleaned up at exit.
_browser = None
_playwright_ctx = None
_browser_lock = asyncio.Lock()


async def _get_browser():
    """Return a reusable Playwright Chromium browser, launching if needed.

    Serialized by _browser_lock to prevent concurrent launches.
    """
    global _browser, _playwright_ctx
    async with _browser_lock:
        if _browser is not None and _browser.is_connected():
            return _browser
        # Clean up stale context if browser crashed
        if _playwright_ctx is not None:
            try:
                if _browser:
                    await _browser.close()
            except Exception:
                pass
            try:
                await _playwright_ctx.stop()
            except Exception:
                pass
            _browser = None
            _playwright_ctx = None
        from playwright.async_api import async_playwright

        _playwright_ctx = await async_playwright().start()
        _browser = await _playwright_ctx.chromium.launch(headless=True)
        return _browser


async def _invalidate_browser():
    """Force browser restart on next _get_browser() call."""
    global _browser, _playwright_ctx
    async with _browser_lock:
        try:
            if _browser:
                await _browser.close()
        except Exception:
            pass
        try:
            if _playwright_ctx:
                await _playwright_ctx.stop()
        except Exception:
            pass
        _browser = None
        _playwright_ctx = None


async def _shutdown_browser():
    global _browser, _playwright_ctx
    try:
        if _browser:
            await _browser.close()
        if _playwright_ctx:
            await _playwright_ctx.stop()
    except Exception:
        pass
    _browser = None
    _playwright_ctx = None


def _atexit_cleanup():
    """Best-effort sync cleanup at interpreter exit."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_shutdown_browser())
        else:
            loop.run_until_complete(_shutdown_browser())
    except Exception:
        pass


atexit.register(_atexit_cleanup)

_MD_TIMEOUT = 8.0  # seconds — fast path, no JS rendering needed


async def _try_markdown(url: str) -> str | None:
    """Try Markdown-for-Agents content negotiation. Returns markdown or None."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=_MD_TIMEOUT,
            headers={"User-Agent": _UA, "Accept": "text/markdown"},
        ) as client:
            resp = await client.get(url)
            content_type = resp.headers.get("content-type", "")
            if "text/markdown" in content_type and resp.status_code == 200:
                return resp.text
    except Exception:
        pass
    return None


def _validate_web_fetch(args: dict) -> tuple[bool, str]:
    url = str(args.get("url", "")).strip()
    if not url:
        return False, "url is required"
    # Auto-upgrade http → https to prevent SSRF and MITM risks.
    if url.startswith("http://"):
        url = "https://" + url[7:]
        args["url"] = url
    if not url.startswith("https://"):
        return False, "url must start with https://"
    if len(url) > 2000:
        return False, "url too long"
    return True, "ok"


@tool(
    name="web_fetch",
    description="Fetch webpage content and return plain text. Supports JS-rendered pages (SPA, finance pages, etc.).",
    risk_level="medium",
    capability=ToolCapability(
        network_access=True,
        max_output_chars=_MAX_OUTPUT_CHARS,
        latency_tier="slow",
        failure_modes=("network_timeout", "content_empty"),
    ),
    validator=_validate_web_fetch,
    parameters={
        "properties": {
            "url": {
                "type": "string",
                "description": "Web URL to fetch (must start with https://; http is auto-upgraded)",
            },
        },
        "required": ["url"],
    },
)
async def web_fetch(url: str, **kwargs) -> str:
    """Fetch URL — try Markdown-for-Agents first, fallback to Playwright."""
    # --- Fast path: Markdown content negotiation ---
    md = await _try_markdown(url)
    if md is not None:
        md = _sanitize(md)
        if len(md) > _MAX_OUTPUT_CHARS:
            md = md[:_MAX_OUTPUT_CHARS] + "\n...(truncated)"
        return md if md.strip() else "[Empty content]"

    # --- Fallback: headless Chromium via Playwright ---
    try:
        from playwright.async_api import async_playwright as _check  # noqa: F401
    except ImportError:
        return (
            "[Error] Playwright is not installed. Run:\n"
            "  pip install 'liagent[browser]' && playwright install chromium"
        )

    text = await _fetch_with_playwright(url)
    if text is None:
        return "[Empty content]"
    return text


def _is_browser_dead(exc: Exception) -> bool:
    """Check if exception indicates the browser/page was closed or crashed."""
    name = type(exc).__name__
    return name in ("TargetClosedError", "BrowserClosedError") or "Target closed" in str(exc)


async def _fetch_with_playwright(url: str) -> str | None:
    """Fetch a page via Playwright with one retry on browser crash."""
    for attempt in range(2):
        try:
            return await _fetch_page(url)
        except Exception as e:
            if attempt == 0 and _is_browser_dead(e):
                await _invalidate_browser()
                continue
            return f"[Fetch error] {e}"
    return "[Fetch error] browser restart failed"


async def _fetch_page(url: str) -> str:
    """Single attempt to fetch a page. Raises on browser/page failure."""
    try:
        browser = await _get_browser()
    except Exception as e:
        raise RuntimeError(f"Browser startup error: {e}") from e

    page = None
    try:
        page = await browser.new_page(user_agent=_UA)
        await page.add_init_script(_STEALTH_JS)
        try:
            await page.goto(url, wait_until=_WAIT_STRATEGY, timeout=_TIMEOUT_MS)
        except Exception:
            # Timeout or navigation error — still try to extract whatever loaded
            pass

        # Let JS settle and render dynamic content
        await page.wait_for_timeout(_JS_SETTLE_MS)

        text = await page.inner_text("body")
    finally:
        if page:
            try:
                await page.close()
            except Exception:
                pass

    # Normalize whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()

    # Sanitize against XML injection
    text = _sanitize(text)

    if len(text) > _MAX_OUTPUT_CHARS:
        text = text[:_MAX_OUTPUT_CHARS] + "\n...(truncated)"
    if not text:
        return "[Empty content]"
    return text
