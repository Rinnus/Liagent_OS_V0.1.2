"""Tests for Discord bot image validation, normalization, and cache."""

import base64
import io
import unittest
from pathlib import Path

from PIL import Image

from liagent.ui.discord_bot import (
    _image_bytes_to_data_url,
    _is_image_attachment,
    _prepare_discord_image,
)
from liagent.ui.vision_routes import _cleanup_path


def _make_jpeg(w: int = 320, h: int = 240, color: tuple = (128, 100, 80)) -> bytes:
    """Create a test JPEG with a gradient (sufficient contrast for validation)."""
    img = Image.new("RGB", (w, h), color)
    # Add horizontal gradient to ensure contrast > 2.0
    pixels = img.load()
    for x in range(w):
        for y in range(min(h, 20)):  # gradient band at top
            r = min(255, color[0] + x % 80)
            g = min(255, color[1] + (x * 2) % 60)
            b = min(255, color[2] + (x * 3) % 50)
            pixels[x, y] = (r, g, b)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def _make_png(w: int = 320, h: int = 240) -> bytes:
    """Create a test PNG with gradient."""
    img = Image.new("RGB", (w, h), (100, 120, 140))
    pixels = img.load()
    for x in range(w):
        for y in range(min(h, 20)):
            pixels[x, y] = (x % 256, (x * 2) % 256, (x * 3) % 256)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ── _is_image_attachment ─────────────────────────────────────────────────

class TestIsImageAttachment(unittest.TestCase):
    def test_jpeg_content_type(self):
        self.assertTrue(_is_image_attachment("image/jpeg", "photo.jpg"))

    def test_png_content_type(self):
        self.assertTrue(_is_image_attachment("image/png", None))

    def test_filename_fallback(self):
        self.assertTrue(_is_image_attachment(None, "screenshot.png"))

    def test_non_image(self):
        self.assertFalse(_is_image_attachment("text/plain", "notes.txt"))

    def test_audio_not_image(self):
        self.assertFalse(_is_image_attachment("audio/ogg", "voice.ogg"))


# ── _image_bytes_to_data_url ─────────────────────────────────────────────

class TestImageBytesToDataUrl(unittest.TestCase):
    def test_jpeg_content_type(self):
        data = _make_jpeg()
        url = _image_bytes_to_data_url(data, content_type="image/jpeg", filename="test.jpg")
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))

    def test_unknown_content_type_uses_filename(self):
        data = _make_png()
        url = _image_bytes_to_data_url(data, content_type=None, filename="test.png")
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_fallback_to_jpeg(self):
        url = _image_bytes_to_data_url(b"raw", content_type=None, filename=None)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))


# ── _prepare_discord_image ───────────────────────────────────────────────

class TestPrepareDiscordImage(unittest.TestCase):
    def test_normal_image_accepted(self):
        """Standard JPEG passes validation and returns data_url."""
        data = _make_jpeg(320, 240)
        url, notes, cache_update = _prepare_discord_image(
            data, content_type="image/jpeg", filename="photo.jpg",
            vision_cache=None,
        )
        self.assertIsNotNone(url)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        # Cache should be updated
        self.assertIsNotNone(cache_update)
        self.assertIn("digest", cache_update)
        self.assertIn("path", cache_update)
        self.assertIn("ts_ms", cache_update)
        # Clean up
        _cleanup_path(cache_update["path"])

    def test_oversized_image_rejected(self):
        """Image exceeding _MAX_IMAGE_BYTES is rejected."""
        from liagent.ui.vision_routes import _MAX_IMAGE_BYTES
        # Create image larger than limit
        huge_data = b"\xff\xd8\xff\xe0" + b"\x00" * (_MAX_IMAGE_BYTES + 1000)
        url, notes, cache_update = _prepare_discord_image(
            huge_data, content_type="image/jpeg", filename="huge.jpg",
            vision_cache=None,
        )
        self.assertIsNone(url)
        self.assertIn("image_too_large", notes)
        self.assertIsNone(cache_update)

    def test_tiny_image_rejected(self):
        """Image smaller than _MIN_IMAGE_SIDE is rejected."""
        data = _make_jpeg(30, 30)
        url, notes, cache_update = _prepare_discord_image(
            data, content_type="image/jpeg", filename="tiny.jpg",
            vision_cache=None,
        )
        self.assertIsNone(url)
        self.assertIn("image_too_small", notes)

    def test_blank_image_rejected(self):
        """Solid-color image (zero contrast) is rejected."""
        # Create a truly blank solid-color image (no gradient)
        img = Image.new("RGB", (200, 200), (128, 128, 128))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=95)
        data = out.getvalue()
        url, notes, cache_update = _prepare_discord_image(
            data, content_type="image/jpeg", filename="blank.jpg",
            vision_cache=None,
        )
        self.assertIsNone(url)
        self.assertIn("blank_image_skipped", notes)

    def test_large_image_normalized(self):
        """Image exceeding _MAX_IMAGE_SIDE is downscaled."""
        data = _make_jpeg(2048, 1536)
        url, notes, cache_update = _prepare_discord_image(
            data, content_type="image/jpeg", filename="big.jpg",
            vision_cache=None,
        )
        self.assertIsNotNone(url)
        self.assertIn("image_normalized", notes)
        # Verify output is smaller than input
        b64_part = url.split(",", 1)[1]
        output_bytes = base64.b64decode(b64_part)
        self.assertLess(len(output_bytes), len(data))
        if cache_update:
            _cleanup_path(cache_update["path"])

    def test_png_converted_to_jpeg(self):
        """PNG input is normalized to JPEG output."""
        data = _make_png(320, 240)
        url, notes, cache_update = _prepare_discord_image(
            data, content_type="image/png", filename="screenshot.png",
            vision_cache=None,
        )
        self.assertIsNotNone(url)
        # Output should be JPEG data URL
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        if cache_update:
            _cleanup_path(cache_update["path"])

    def test_dedup_same_image(self):
        """Sending the same image twice returns cached version."""
        data = _make_jpeg(320, 240)

        # First send — creates cache
        url1, notes1, cache1 = _prepare_discord_image(
            data, content_type="image/jpeg", filename="photo.jpg",
            vision_cache=None,
        )
        self.assertIsNotNone(url1)
        self.assertIsNotNone(cache1)

        # Second send with same image — should hit dedup
        url2, notes2, cache2 = _prepare_discord_image(
            data, content_type="image/jpeg", filename="photo.jpg",
            vision_cache=cache1,
        )
        self.assertIsNotNone(url2)
        self.assertIn("vision_frame_unchanged", notes2)
        # No new cache update needed
        self.assertIsNone(cache2)

        # Clean up
        _cleanup_path(cache1["path"])

    def test_different_image_updates_cache(self):
        """Different image replaces cache."""
        data1 = _make_jpeg(320, 240, color=(100, 80, 60))
        data2 = _make_jpeg(320, 240, color=(200, 180, 160))

        url1, _, cache1 = _prepare_discord_image(
            data1, content_type="image/jpeg", filename="a.jpg",
            vision_cache=None,
        )
        self.assertIsNotNone(cache1)

        url2, _, cache2 = _prepare_discord_image(
            data2, content_type="image/jpeg", filename="b.jpg",
            vision_cache=cache1,
        )
        self.assertIsNotNone(url2)
        self.assertIsNotNone(cache2)
        # Digest should be different
        self.assertNotEqual(cache1["digest"], cache2["digest"])

        _cleanup_path(cache1["path"])
        _cleanup_path(cache2["path"])

    def test_expired_cache_not_used_for_dedup(self):
        """Expired cache entry is not used for deduplication."""
        data = _make_jpeg(320, 240)
        _, _, cache = _prepare_discord_image(
            data, content_type="image/jpeg", filename="photo.jpg",
            vision_cache=None,
        )
        self.assertIsNotNone(cache)

        # Expire the cache by setting ts_ms far in the past
        cache["ts_ms"] = 0

        url2, notes2, cache2 = _prepare_discord_image(
            data, content_type="image/jpeg", filename="photo.jpg",
            vision_cache=cache,
        )
        self.assertIsNotNone(url2)
        # Should NOT say "unchanged" since cache is expired
        if notes2:
            self.assertNotIn("vision_frame_unchanged", notes2)
        self.assertIsNotNone(cache2)  # New cache created

        _cleanup_path(cache["path"])
        if cache2:
            _cleanup_path(cache2["path"])
