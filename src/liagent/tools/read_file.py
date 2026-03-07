"""File reading tool — reads text files and PDFs from the cwork sandbox."""

from . import ToolCapability, tool
from ._path_security import _validate_cwork_path

_MAX_OUTPUT_CHARS = 4000
_PDF_MAX_PAGES_DEFAULT = 20  # pages per call (keeps output within LLM context)


def _validate_read_file(args: dict) -> tuple[bool, str]:
    path = str(args.get("path", "")).strip()
    ok, reason, _ = _validate_cwork_path(path)
    return ok, reason


def _read_pdf(resolved, start_page: int = 1, max_pages: int = _PDF_MAX_PAGES_DEFAULT) -> str:
    """Extract text from a PDF file using pdfplumber. Returns text with page markers."""
    try:
        import pdfplumber
    except ImportError:
        return (
            "[PDF error] pdfplumber is not installed. Run:\n"
            "  pip install pdfplumber"
        )

    try:
        with pdfplumber.open(resolved) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                return "[Empty PDF]"

            # Clamp page range
            start_idx = max(0, start_page - 1)
            end_idx = min(total_pages, start_idx + max_pages)

            parts = []
            chars = 0
            for i in range(start_idx, end_idx):
                page_text = pdf.pages[i].extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue
                marker = f"[Page {i + 1}/{total_pages}]"
                parts.append(f"{marker}\n{page_text}")
                chars += len(page_text) + len(marker) + 2
                if chars >= _MAX_OUTPUT_CHARS:
                    break

            if not parts:
                return f"[PDF] {total_pages} pages, no extractable text in pages {start_page}-{end_idx}"

            content = "\n\n".join(parts)
            if len(content) > _MAX_OUTPUT_CHARS:
                content = content[:_MAX_OUTPUT_CHARS] + "\n...(truncated)"

            # Append navigation hint if there are more pages
            if end_idx < total_pages:
                content += (
                    f"\n\n[INFO] Showing pages {start_page}-{end_idx} of {total_pages}. "
                    f"Call read_file with start_page={end_idx + 1} to continue."
                )
            return content
    except Exception as e:
        return f"[PDF error] {e}"


@tool(
    name="read_file",
    description=(
        "Read content from a file path. Supports text files (UTF-8) and PDF files. "
        "For large PDFs, use start_page to read specific sections. "
        "Requires an explicit full path. Use `list_dir` to inspect directory contents."
    ),
    risk_level="medium",
    capability=ToolCapability(
        filesystem_access=True,
        max_output_chars=_MAX_OUTPUT_CHARS,
    ),
    validator=_validate_read_file,
    parameters={
        "properties": {
            "path": {
                "type": "string",
                "description": "File path (must be inside cwork)",
            },
            "start_page": {
                "type": "integer",
                "description": "Starting page number for PDF files (1-based, default: 1)",
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum pages to read from PDF (default: 20)",
            },
        },
        "required": ["path"],
    },
)
async def read_file(path: str, start_page: int = 1, max_pages: int = _PDF_MAX_PAGES_DEFAULT, **kwargs) -> str:
    """Read a text or PDF file from the cwork sandbox."""
    ok, reason, resolved = _validate_cwork_path(path)
    if not ok or resolved is None:
        return f"[Path error] {reason}"

    if not resolved.exists():
        return f"[Not found] {resolved}"

    if not resolved.is_file():
        return f"[Not a file] {resolved}"

    # PDF detection
    if resolved.suffix.lower() == ".pdf":
        return _read_pdf(
            resolved,
            start_page=max(1, int(start_page or 1)),
            max_pages=max(1, min(50, int(max_pages or _PDF_MAX_PAGES_DEFAULT))),
        )

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"[Encoding error] File is not UTF-8 text: {resolved.name}"
    except Exception as e:
        return f"[Read error] {e}"

    if len(content) > _MAX_OUTPUT_CHARS:
        content = content[:_MAX_OUTPUT_CHARS] + "\n...(truncated)"
    if not content:
        return "[Empty file]"
    return content
