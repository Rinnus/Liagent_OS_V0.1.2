"""Vision/image processing utilities for the web server.

Contains image decoding, normalization, caching, and resolution helpers
used by the WebSocket chat handler.
"""

import base64
import binascii
import hashlib
import io
import os
import tempfile
import time
from pathlib import Path

_MAX_IMAGE_AGE_MS = max(500, int(os.environ.get("LIAGENT_MAX_IMAGE_AGE_MS", "5000")))
_MAX_IMAGE_BYTES = max(
    64 * 1024,
    int(os.environ.get("LIAGENT_MAX_IMAGE_BYTES", str(3 * 1024 * 1024))),
)
_MIN_IMAGE_SIDE = max(32, int(os.environ.get("LIAGENT_MIN_IMAGE_SIDE", "96")))
_MAX_IMAGE_SIDE = max(256, int(os.environ.get("LIAGENT_MAX_IMAGE_SIDE", "1024")))
_IMAGE_JPEG_QUALITY = max(
    45, min(95, int(os.environ.get("LIAGENT_IMAGE_JPEG_QUALITY", "72")))
)
_IMAGE_CONTEXT_TTL_MS = max(
    _MAX_IMAGE_AGE_MS, int(os.environ.get("LIAGENT_IMAGE_CONTEXT_TTL_MS", "180000"))
)
_CAMERA_REUSE_GRACE_MS = max(
    0, int(os.environ.get("LIAGENT_CAMERA_REUSE_GRACE_MS", "3000"))
)


def _cleanup_path(path: str | None):
    if not path:
        return
    try:
        p = Path(path)
        if p.exists():
            p.unlink(missing_ok=True)
    except Exception:
        pass


def _clone_image_path(src: str) -> str | None:
    try:
        data = Path(src).read_bytes()
        suffix = Path(src).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(data)
        tmp.close()
        return tmp.name
    except Exception:
        return None


def _cache_valid(cache: dict | None, now_ms: int) -> bool:
    if not cache:
        return False
    path = str(cache.get("path", "")).strip()
    ts_ms = int(cache.get("ts_ms", 0) or 0)
    if not path or ts_ms <= 0:
        return False
    if now_ms - ts_ms > _IMAGE_CONTEXT_TTL_MS:
        return False
    return Path(path).exists()


def _decode_and_prepare_image(
    image_b64: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Decode image b64, normalize resolution/format, and return temp path + digest."""
    try:
        payload = image_b64.strip()
        if "," in payload and payload[:32].lower().startswith("data:"):
            header, payload = payload.split(",", 1)
            hl = header.lower()
        else:
            hl = ""
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error):
        return None, None, None, "image_decode_failed"
    except Exception:
        return None, None, None, "image_decode_failed"

    if len(raw) > _MAX_IMAGE_BYTES:
        return None, None, None, f"image_too_large(bytes={len(raw)})"

    try:
        from PIL import Image, ImageStat

        img = Image.open(io.BytesIO(raw))
        img.load()
        if img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")
        w, h = img.size
    except Exception:
        return None, None, None, "image_open_failed"

    if min(w, h) < _MIN_IMAGE_SIDE:
        return None, None, None, f"image_too_small({w}x{h})"

    notes: list[str] = []
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    contrast = float(stat.stddev[0]) if stat.stddev else 0.0
    mean_luma = float(stat.mean[0]) if stat.mean else 0.0
    if contrast < 2.0:
        return None, None, None, "blank_image_skipped"
    if contrast < 8.0:
        notes.append("low_contrast_frame")
    if mean_luma < 20.0:
        notes.append("very_dark_frame")
    elif mean_luma > 235.0:
        notes.append("overexposed_frame")

    max_side = max(w, h)
    if max_side > _MAX_IMAGE_SIDE:
        scale = float(_MAX_IMAGE_SIDE) / float(max_side)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        notes.append(f"image_normalized({w}x{h}->{nw}x{nh})")

    try:
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=_IMAGE_JPEG_QUALITY, optimize=True)
        out_bytes = out.getvalue()
        digest = hashlib.sha1(out_bytes).hexdigest()
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(out_bytes)
        tmp.close()
        if not notes and ("image/jpeg" in hl or "image/jpg" in hl):
            notes.append("image_ok")
        return tmp.name, digest, "; ".join(notes) if notes else None, None
    except Exception:
        return None, None, None, "image_normalize_failed"


def _resolve_image_paths(
    payload: dict,
    *,
    vision_cache: dict | None,
) -> tuple[list[str] | None, str | None, dict | None]:
    """Resolve vision images with freshness checks and optional camera cache reuse."""
    now_ms = int(time.time() * 1000)
    image_b64 = payload.get("image")
    allow_reuse = bool(payload.get("reuse_vision", False))
    if not image_b64:
        if allow_reuse and _cache_valid(vision_cache, now_ms):
            cloned = _clone_image_path(str(vision_cache["path"]))
            if cloned:
                return [cloned], "vision_reused_from_cache", None
        return None, None, None

    notes: list[str] = []
    age = payload.get("image_age_ms")
    if age is not None:
        try:
            age_f = float(age)
            if age_f > _MAX_IMAGE_AGE_MS:
                notes.append(f"stale_image_skipped(age_ms={int(age_f)})")
                if (
                    allow_reuse
                    and age_f <= (_MAX_IMAGE_AGE_MS + _CAMERA_REUSE_GRACE_MS)
                    and _cache_valid(vision_cache, now_ms)
                ):
                    cloned = _clone_image_path(str(vision_cache["path"]))
                    if cloned:
                        notes.append("vision_reused_from_cache")
                        return [cloned], "; ".join(notes), None
                return None, "; ".join(notes), None
        except (TypeError, ValueError):
            pass

    run_path, digest, prep_note, prep_err = _decode_and_prepare_image(str(image_b64))
    if prep_err:
        notes.append(prep_err)
        if allow_reuse and _cache_valid(vision_cache, now_ms):
            cloned = _clone_image_path(str(vision_cache["path"]))
            if cloned:
                notes.append("vision_reused_from_cache")
                return [cloned], "; ".join(notes), None
        return None, "; ".join(notes), None
    if prep_note:
        notes.append(prep_note)

    cache_update = None
    if run_path:
        if _cache_valid(vision_cache, now_ms) and digest and vision_cache.get("digest") == digest:
            _cleanup_path(run_path)
            cloned = _clone_image_path(str(vision_cache["path"]))
            if cloned:
                notes.append("vision_frame_unchanged")
                return [cloned], "; ".join(notes) if notes else None, None
        cache_copy = _clone_image_path(run_path)
        if cache_copy:
            cache_update = {
                "path": cache_copy,
                "digest": digest or "",
                "ts_ms": now_ms,
            }

    return ([run_path] if run_path else None), ("; ".join(notes) if notes else None), cache_update
