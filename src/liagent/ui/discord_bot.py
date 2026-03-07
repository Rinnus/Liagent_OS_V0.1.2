"""Discord Bot thin client — forwards messages to LiAgent WebSocket backend.

Requires: pip install 'liagent[discord]'
Usage:    LIAGENT_DISCORD_TOKEN=... liagent --discord

The bot connects to ws://localhost:8080/ws/chat and relays text/voice.
It holds no models — all inference is done by the LiAgent web server.
"""

import asyncio
import base64
import io
import json
import mimetypes
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    import discord
    from discord.ext import commands
except ImportError:
    raise ImportError(
        "py-cord is required for Discord bot. Install with: "
        "pip install 'liagent[discord]'"
    )

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    raise ImportError(
        "websockets is required for Discord bot. Install with: "
        "pip install 'liagent[discord]'"
    )

from ..logging import get_logger

_log = get_logger("discord_bot")

_AUDIO_EXTS = (".ogg", ".mp3", ".wav", ".m4a")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
_CONFIRM_TIMEOUT = 120  # seconds before confirmation buttons expire


def _is_audio_attachment(content_type: str | None, filename: str | None) -> bool:
    ct = (content_type or "").lower()
    fn = (filename or "").lower()
    return ("audio" in ct) or fn.endswith(_AUDIO_EXTS)


def _is_image_attachment(content_type: str | None, filename: str | None) -> bool:
    ct = (content_type or "").lower()
    fn = (filename or "").lower()
    return ct.startswith("image/") or fn.endswith(_IMAGE_EXTS)


def _image_bytes_to_data_url(
    image_bytes: bytes,
    *,
    content_type: str | None,
    filename: str | None,
) -> str:
    """Encode attachment bytes to a data URL accepted by vision_routes."""
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if not mime.startswith("image/"):
        guessed, _ = mimetypes.guess_type(filename or "")
        if guessed and guessed.startswith("image/"):
            mime = guessed.lower()
        else:
            mime = "image/jpeg"
    payload = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{payload}"


def _prepare_discord_image(
    image_bytes: bytes,
    *,
    content_type: str | None,
    filename: str | None,
    vision_cache: dict | None,
) -> tuple[str | None, str | None, dict | None]:
    """Validate, normalize, and deduplicate an image from Discord.

    Uses the same vision pipeline as the web server:
    - Size check (reject > _MAX_IMAGE_BYTES)
    - PIL validation, contrast/quality analysis
    - Resolution normalization (≤1024px)
    - JPEG compression (Q72)
    - SHA1 digest dedup against vision_cache

    Returns: (data_url | None, notes | None, cache_update | None)
    """
    import time
    from .vision_routes import (
        _MAX_IMAGE_BYTES,
        _IMAGE_CONTEXT_TTL_MS,
        _cache_valid,
        _cleanup_path,
        _clone_image_path,
        _decode_and_prepare_image,
    )

    now_ms = int(time.time() * 1000)
    notes: list[str] = []

    # Pre-check raw size before base64 encoding
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        notes.append(f"image_too_large({len(image_bytes)} bytes, max {_MAX_IMAGE_BYTES})")
        return None, "; ".join(notes), None

    # Convert to data URL for the vision pipeline
    raw_data_url = _image_bytes_to_data_url(
        image_bytes, content_type=content_type, filename=filename,
    )

    # Run through the same validation/normalization pipeline as web server
    run_path, digest, prep_note, prep_err = _decode_and_prepare_image(raw_data_url)
    if prep_err:
        notes.append(prep_err)
        return None, "; ".join(notes), None
    if prep_note:
        notes.append(prep_note)

    # Dedup: check if this image is identical to the cached one
    if _cache_valid(vision_cache, now_ms) and digest and vision_cache.get("digest") == digest:
        _cleanup_path(run_path)
        notes.append("vision_frame_unchanged")
        # Re-read cached file for the data URL
        cached_path = str(vision_cache["path"])
        try:
            cached_bytes = open(cached_path, "rb").read()
            b64 = base64.b64encode(cached_bytes).decode()
            return f"data:image/jpeg;base64,{b64}", "; ".join(notes), None
        except Exception:
            pass  # Fall through to use run_path

    # Read normalized file and build data URL
    if not run_path:
        return None, "; ".join(notes) if notes else None, None

    try:
        normalized_bytes = open(run_path, "rb").read()
        b64 = base64.b64encode(normalized_bytes).decode()
        data_url = f"data:image/jpeg;base64,{b64}"
    except Exception:
        _cleanup_path(run_path)
        notes.append("image_read_failed")
        return None, "; ".join(notes), None

    # Update cache
    cache_update = None
    cache_copy = _clone_image_path(run_path)
    if cache_copy:
        cache_update = {
            "path": cache_copy,
            "digest": digest or "",
            "ts_ms": now_ms,
        }

    _cleanup_path(run_path)
    return data_url, "; ".join(notes) if notes else None, cache_update


# ─── Audio transcoding helpers (FFmpeg) ─────────────────────────────────────

def _pcm_48k_stereo_to_16k_mono_f32(pcm_bytes: bytes) -> bytes:
    """Convert Discord PCM (48kHz stereo s16le) to STT format (16kHz mono float32).

    Uses FFmpeg for sample rate conversion and channel downmix.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ar", "48000", "-ac", "2", "-i", "pipe:0",
            "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1",
        ],
        input=pcm_bytes,
        capture_output=True,
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg resample failed: {proc.stderr.decode()[:200]}")
    return proc.stdout


def _f32_to_ogg_opus(f32_bytes: bytes, src_rate: int) -> bytes:
    """Convert float32 mono audio to OGG/Opus for Discord voice message reply."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(src_rate), "-ac", "1", "-i", "pipe:0",
            "-c:a", "libopus", "-b:a", "64k", "-f", "ogg", "pipe:1",
        ],
        input=f32_bytes,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg ogg encode failed: {proc.stderr.decode()[:200]}")
    return proc.stdout


def _ogg_to_16k_mono_f32(ogg_bytes: bytes) -> bytes:
    """Convert OGG/Opus voice message to STT format (16kHz mono float32)."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1",
        ],
        input=ogg_bytes,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg ogg decode failed: {proc.stderr.decode()[:200]}")
    return proc.stdout


def _f32_to_48k_stereo_s16le(audio_f32: np.ndarray, src_rate: int) -> bytes:
    """Convert TTS output (mono float32 at src_rate) to Discord PCM (48kHz stereo s16le).

    Uses FFmpeg for sample rate conversion and channel upmix.
    """
    raw = audio_f32.astype(np.float32).tobytes()
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(src_rate), "-ac", "1", "-i", "pipe:0",
            "-f", "s16le", "-ar", "48000", "-ac", "2", "pipe:1",
        ],
        input=raw,
        capture_output=True,
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg resample failed: {proc.stderr.decode()[:200]}")
    return proc.stdout


# ─── Energy-based VAD ───────────────────────────────────────────────────────

class EnergyVAD:
    """Simple energy-based voice activity detector for Discord PCM input.

    Accumulates PCM frames and detects speech segments based on
    RMS energy threshold and silence duration.
    """

    def __init__(
        self,
        *,
        energy_threshold: float = 300.0,
        silence_duration: float = 1.5,
        sample_rate: int = 48000,
        channels: int = 2,
    ):
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate * channels)
        self._buffer = bytearray()
        self._speech_started = False
        self._silence_count = 0

    def feed(self, pcm_bytes: bytes) -> bytes | None:
        """Feed PCM data. Returns complete speech segment or None."""
        # Calculate RMS energy of this chunk
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if samples.size == 0:
            return None
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

        if rms >= self.energy_threshold:
            self._speech_started = True
            self._silence_count = 0
            self._buffer.extend(pcm_bytes)
        elif self._speech_started:
            self._silence_count += len(pcm_bytes)
            self._buffer.extend(pcm_bytes)
            if self._silence_count >= self.silence_samples:
                # Speech segment complete
                segment = bytes(self._buffer)
                self._buffer.clear()
                self._speech_started = False
                self._silence_count = 0
                return segment

        return None

    def reset(self):
        self._buffer.clear()
        self._speech_started = False
        self._silence_count = 0


# ─── Voice Sink (py-cord recording) ────────────────────────────────────────

class LiAgentVoiceSink(discord.sinks.Sink):
    """Captures voice from Discord, runs VAD, sends speech to LiAgent WS.

    IMPORTANT: py-cord calls write() from a decoder thread, not the async
    event loop. We use run_coroutine_threadsafe to bridge back to async.
    """

    def __init__(self, bot: "LiAgentBot", text_channel, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.bot = bot
        self.text_channel = text_channel
        self._loop = loop
        self._vads: dict[int, EnergyVAD] = {}  # user_id -> VAD
        self._processing = False  # simple guard against overlapping segments

    def write(self, data: bytes, user: int):
        """Called by py-cord from a THREAD with raw PCM data per user."""
        if user not in self._vads:
            self._vads[user] = EnergyVAD()
        segment = self._vads[user].feed(data)
        if segment and not self._processing:
            self._processing = True
            # Bridge from thread → async event loop
            asyncio.run_coroutine_threadsafe(
                self._process_segment(segment, user), self._loop
            )

    async def _process_segment(self, pcm_bytes: bytes, user_id: int):
        """Convert speech segment and send to LiAgent for STT + response."""
        try:
            print(f"  [voice] speech detected ({len(pcm_bytes)} bytes), processing...")

            # Convert Discord PCM to 16kHz mono float32 for STT
            loop = asyncio.get_event_loop()
            f32_bytes = await loop.run_in_executor(
                None, _pcm_48k_stereo_to_16k_mono_f32, pcm_bytes
            )
            audio_b64 = base64.b64encode(f32_bytes).decode()
            print(f"  [voice] converted to f32 ({len(f32_bytes)} bytes), sending to LiAgent...")
            session_key = self.bot._session_key_for_channel(
                self.text_channel, user_id=user_id
            )

            # Send audio to LiAgent WS
            reply = await self.bot.send_to_liagent(
                {"type": "audio", "audio": audio_b64, "session_key": session_key}
            )

            # Collect TTS audio chunks for playback
            tts_chunks: list[tuple[np.ndarray, int]] = []
            stt_text = ""
            final_text = ""

            for msg in reply:
                mtype = msg.get("type", "")
                if mtype == "stt_result":
                    stt_text = msg.get("text", "")
                    print(f"  [voice] STT: {stt_text}")
                elif mtype == "tts_chunk":
                    audio_data = base64.b64decode(msg["audio"])
                    audio_np = np.frombuffer(audio_data, dtype=np.float32)
                    sr = msg.get("sample_rate", 24000)
                    tts_chunks.append((audio_np, sr))
                elif mtype == "done":
                    final_text = msg.get("text", "")
                elif mtype == "error":
                    err = msg.get("text", "")
                    print(f"  [voice] error from LiAgent: {err}")
                    await self.text_channel.send(f"Error: {err}")
                    return

            # Play TTS audio in voice channel
            vc = self.bot.voice_clients[0] if self.bot.voice_clients else None
            if vc and vc.is_connected() and tts_chunks:
                combined = np.concatenate([c[0] for c in tts_chunks])
                sr = tts_chunks[0][1]
                print(f"  [voice] playing TTS ({combined.size} samples @ {sr}Hz)...")
                pcm_out = await loop.run_in_executor(
                    None, _f32_to_48k_stereo_s16le, combined, sr
                )
                source = discord.PCMAudio(io.BytesIO(pcm_out))
                if vc.is_playing():
                    vc.stop()
                vc.play(source)
            elif tts_chunks:
                print("  [voice] no voice client connected, skipping TTS playback")

            # Also send text to channel for reference
            if final_text:
                for i in range(0, len(final_text), 2000):
                    await self.text_channel.send(final_text[i : i + 2000])

        except Exception as e:
            print(f"  [voice] error: {e}")
            try:
                await self.text_channel.send(f"Voice error: {e}")
            except Exception:
                pass
        finally:
            self._processing = False

    def cleanup(self):
        self._vads.clear()


# ─── Push channel config ─────────────────────────────────────────────────────

def _get_push_channel_id() -> int | None:
    raw = os.environ.get("LIAGENT_DISCORD_PUSH_CHANNEL", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return None


def _get_api_base(ws_url: str) -> str:
    """Derive HTTP API base URL from WebSocket URL."""
    base = ws_url.replace("ws://", "http://").replace("wss://", "https://")
    return base.split("/ws/")[0]


def _build_discord_session_key(
    *,
    guild_id: int | None,
    channel_id: int | None,
    user_id: int | None,
    parent_channel_id: int | None = None,
) -> str:
    """Build a stable session key for per-user REPL isolation in Discord."""
    gid = str(guild_id) if guild_id is not None else "dm"
    cid = str(channel_id) if channel_id is not None else "unknown"
    uid = str(user_id) if user_id is not None else "anon"
    key = f"discord:g:{gid}:c:{cid}:u:{uid}"
    if parent_channel_id is not None:
        key += f":p:{parent_channel_id}"
    return key


# ─── Tool confirmation View ──────────────────────────────────────────────────

class _ConfirmationView(discord.ui.View):
    """Discord UI View with Approve / Reject buttons for tool confirmation."""

    def __init__(
        self,
        bot: "LiAgentBot",
        token: str,
        tool: str,
        brief: str,
        *,
        session_key: str | None = None,
    ):
        super().__init__(timeout=_CONFIRM_TIMEOUT)
        self.bot = bot
        self.token = token
        self.tool = tool
        self.brief = brief
        self.session_key = session_key
        self.result: dict | None = None
        self._done = asyncio.Event()

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green, emoji="\u2705")
    async def approve(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._resolve(interaction, approved=True, force=False)

    @discord.ui.button(label="Reject", style=discord.ButtonStyle.red, emoji="\u274c")
    async def reject(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._resolve(interaction, approved=False, force=False)

    async def _resolve(self, interaction: discord.Interaction, *, approved: bool, force: bool):
        # Guard: ignore duplicate clicks after resolution
        if self._done.is_set():
            return
        for child in self.children:
            child.disabled = True
        try:
            await interaction.response.edit_message(view=self)
        except discord.NotFound:
            # Interaction expired (10062) — buttons are stale, still proceed
            pass
        except Exception:
            pass

        try:
            payload = {
                "type": "tool_confirm",
                "token": self.token,
                "approved": approved,
                "force": force,
            }
            if self.session_key:
                payload["session_key"] = self.session_key
            events = await self.bot.send_to_liagent(payload)
            # Find the tool_confirm_result
            for ev in events:
                if ev.get("type") == "tool_confirm_result":
                    self.result = ev.get("result", {})
                    break
            if self.result is None:
                self.result = {"status": "ok"}

            # Handle need_second_confirm — replace with a force-confirm view
            if self.result.get("status") == "need_second_confirm":
                new_token = self.result.get("token", self.token)
                view2 = _SecondConfirmView(
                    self.bot,
                    new_token,
                    self.tool,
                    session_key=self.session_key,
                )
                try:
                    await interaction.followup.send(
                        f"**Second confirmation required** for `{self.tool}`\n{self.result.get('message', '')}",
                        view=view2,
                    )
                except discord.NotFound:
                    self.result = {"status": "error", "message": "interaction expired"}
                    self._done.set()
                    return
                # Wait for second-stage resolution
                await view2.wait_for_result()
                self.result = view2.result or {"status": "timeout"}
        except Exception as e:
            self.result = {"status": "error", "message": str(e)}
        self._done.set()

    async def wait_for_result(self) -> dict | None:
        """Block until a button is pressed or timeout."""
        try:
            await asyncio.wait_for(self._done.wait(), timeout=_CONFIRM_TIMEOUT)
        except asyncio.TimeoutError:
            self.result = {"status": "timeout", "message": "Confirmation timed out."}
        return self.result

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        self._done.set()


class _ProactiveSuggestionView(discord.ui.View):
    """Discord UI View with Accept / Dismiss buttons for proactive suggestions."""

    def __init__(
        self,
        bot: "LiAgentBot",
        suggestion_id: int | str,
        *,
        session_key: str | None = None,
    ):
        super().__init__(timeout=300)  # 5 minutes for non-urgent suggestions
        self.bot = bot
        self.suggestion_id = suggestion_id
        self.session_key = session_key
        self._resolved = False

    @discord.ui.button(label="Accept", style=discord.ButtonStyle.green, emoji="\u2705")
    async def accept(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._resolve(interaction, action="accept")

    @discord.ui.button(label="Dismiss", style=discord.ButtonStyle.secondary, emoji="\U0001f6ab")
    async def dismiss(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._resolve(interaction, action="dismiss")

    async def _send_feedback(self, action: str):
        """Send suggestion feedback as fire-and-forget (no terminal event expected)."""
        try:
            await self.bot._ensure_ws()
            async with self.bot._ws_lock:
                payload = {
                    "type": "suggestion_feedback",
                    "id": self.suggestion_id,
                    "action": action,
                }
                session_key = self.bot._remember_session_key(self.session_key)
                if session_key:
                    payload["session_key"] = session_key
                await self.bot._ws.send(json.dumps(payload))
        except Exception:
            pass

    async def _resolve(self, interaction: discord.Interaction, *, action: str):
        if self._resolved:
            return
        self._resolved = True
        for child in self.children:
            child.disabled = True
        try:
            await interaction.response.edit_message(view=self)
        except (discord.NotFound, Exception):
            pass
        await self._send_feedback(action)

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        if not self._resolved:
            self._resolved = True
            await self._send_feedback("dismiss")


class _HeartbeatConfirmView(discord.ui.View):
    """Approve / Reject buttons for heartbeat high-risk actions."""

    def __init__(self, bot: "LiAgentBot", token: str, description: str,
                 expires_at: str, api_base: str, api_secret: str = ""):
        # Calculate timeout from expires_at
        try:
            from datetime import datetime, timezone
            exp = datetime.fromisoformat(expires_at)
            now = datetime.now(timezone.utc)
            remaining = max(15, (exp - now).total_seconds())
        except (ValueError, TypeError):
            remaining = 300
        super().__init__(timeout=remaining)
        self.bot = bot
        self.token = token
        self.description = description
        self.api_base = api_base
        self.api_secret = api_secret
        self._resolved = False

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green, emoji="\u2705")
    async def approve(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._resolve(interaction, approved=True)

    @discord.ui.button(label="Reject", style=discord.ButtonStyle.red, emoji="\u274c")
    async def reject(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._resolve(interaction, approved=False)

    async def _resolve(self, interaction: discord.Interaction, *, approved: bool):
        if self._resolved:
            return
        self._resolved = True
        for child in self.children:
            child.disabled = True
        try:
            await interaction.response.edit_message(view=self)
        except (discord.NotFound, Exception):
            pass

        # Call REST endpoint
        import aiohttp
        status_text = "approved" if approved else "rejected"
        headers = {}
        if self.api_secret:
            headers["x-liagent-token"] = self.api_secret
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/api/tasks/confirm",
                    json={"token": self.token, "approved": approved},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    result = await resp.json()
                    if result.get("error") == "already_resolved":
                        status_text = "already resolved"
        except Exception as e:
            status_text = f"error: {e}"

        try:
            await interaction.followup.send(
                f"Heartbeat action: **{status_text}** — {self.description[:100]}"
            )
        except (discord.NotFound, Exception):
            pass

    async def on_timeout(self):
        if not self._resolved:
            self._resolved = True
            for child in self.children:
                child.disabled = True


class _SecondConfirmView(discord.ui.View):
    """Second-stage confirmation with --force semantics."""

    def __init__(
        self,
        bot: "LiAgentBot",
        token: str,
        tool: str,
        *,
        session_key: str | None = None,
    ):
        super().__init__(timeout=_CONFIRM_TIMEOUT)
        self.bot = bot
        self.token = token
        self.tool = tool
        self.session_key = session_key
        self.result: dict | None = None
        self._done = asyncio.Event()

    @discord.ui.button(label="Force Approve", style=discord.ButtonStyle.danger, emoji="\u26a0\ufe0f")
    async def force_approve(self, button: discord.ui.Button, interaction: discord.Interaction):
        if self._done.is_set():
            return
        for child in self.children:
            child.disabled = True
        try:
            await interaction.response.edit_message(view=self)
        except discord.NotFound:
            pass
        except Exception:
            pass
        try:
            payload = {
                "type": "tool_confirm",
                "token": self.token,
                "approved": True,
                "force": True,
            }
            if self.session_key:
                payload["session_key"] = self.session_key
            events = await self.bot.send_to_liagent(payload)
            for ev in events:
                if ev.get("type") == "tool_confirm_result":
                    self.result = ev.get("result", {})
                    break
            if self.result is None:
                self.result = {"status": "ok"}
        except Exception as e:
            self.result = {"status": "error", "message": str(e)}
        self._done.set()

    @discord.ui.button(label="Reject", style=discord.ButtonStyle.secondary, emoji="\u274c")
    async def reject(self, button: discord.ui.Button, interaction: discord.Interaction):
        if self._done.is_set():
            return
        for child in self.children:
            child.disabled = True
        try:
            await interaction.response.edit_message(view=self)
        except discord.NotFound:
            pass
        except Exception:
            pass
        try:
            payload = {
                "type": "tool_confirm",
                "token": self.token,
                "approved": False,
                "force": False,
            }
            if self.session_key:
                payload["session_key"] = self.session_key
            await self.bot.send_to_liagent(payload)
        except Exception:
            pass
        self.result = {"status": "rejected"}
        self._done.set()

    async def wait_for_result(self) -> dict | None:
        try:
            await asyncio.wait_for(self._done.wait(), timeout=_CONFIRM_TIMEOUT)
        except asyncio.TimeoutError:
            self.result = {"status": "timeout"}
        return self.result

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        self._done.set()


# ─── Main Bot class ─────────────────────────────────────────────────────────

class LiAgentBot(commands.Bot):
    """Discord Bot that proxies messages to LiAgent WebSocket backend."""

    def __init__(self, ws_url: str, ws_secret: str):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        super().__init__(command_prefix="!", intents=intents)
        self.ws_url = ws_url
        self.ws_secret = ws_secret
        self._ws: Any | None = None
        self._ws_lock = asyncio.Lock()
        self._response_lock = asyncio.Lock()
        self._push_channel_id: int | None = _get_push_channel_id()
        self._push_task: asyncio.Task | None = None
        self._push_ws: Any | None = None
        self._push_ws_lock = asyncio.Lock()
        self._vision_cache: dict | None = None  # SHA1 dedup cache for images
        self._known_session_keys: set[str] = set()
        self._session_channels: dict[str, int] = {}

    def _remember_session_key(self, session_key: str | None) -> str | None:
        key = str(session_key or "").strip()
        if key:
            is_new = key not in self._known_session_keys
            self._known_session_keys.add(key)
            if is_new:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and self._push_ws is not None:
                    loop.create_task(self._sync_push_session_subscriptions())
            return key
        return None

    async def _sync_push_session_subscriptions(self) -> None:
        ws = self._push_ws
        if ws is None:
            return
        payload = {
            "type": "subscribe_sessions",
            "session_keys": sorted(self._known_session_keys),
        }
        try:
            async with self._push_ws_lock:
                if self._push_ws is ws:
                    await ws.send(json.dumps(payload))
        except Exception:
            pass

    def _remember_session_route(self, session_key: str | None, channel_id: int | None) -> str | None:
        key = self._remember_session_key(session_key)
        if key and channel_id is not None:
            self._session_channels[key] = int(channel_id)
        return key

    def _session_key_for_channel(self, channel, *, user_id: int | None) -> str:
        guild = getattr(channel, "guild", None)
        guild_id = getattr(guild, "id", None)
        channel_id = getattr(channel, "id", None)
        parent_channel_id = getattr(channel, "parent_id", None)
        return self._remember_session_route(_build_discord_session_key(
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            parent_channel_id=parent_channel_id,
        ), channel_id) or ""

    def _session_key_for_message(self, message: discord.Message) -> str:
        return self._session_key_for_channel(
            message.channel,
            user_id=getattr(message.author, "id", None),
        )

    def _session_key_for_ctx(self, ctx: discord.ApplicationContext) -> str:
        channel_id = getattr(ctx, "channel_id", None)
        return self._remember_session_route(_build_discord_session_key(
            guild_id=getattr(getattr(ctx, "guild", None), "id", None),
            channel_id=channel_id,
            user_id=getattr(getattr(ctx, "author", None), "id", None),
            parent_channel_id=getattr(getattr(ctx, "channel", None), "parent_id", None),
        ), channel_id) or ""

    # ── WebSocket connection management ──────────────────────────────────

    async def _ensure_ws(self):
        """Connect (or reconnect) to LiAgent WebSocket."""
        if self._ws is not None:
            try:
                # Quick liveness check
                await self._ws.ping()
                return
            except Exception:
                self._ws = None

        ws = await websockets.connect(
            self.ws_url,
            ping_interval=30,
            close_timeout=5,
            max_size=10 * 1024 * 1024,  # 10MB for voice messages
        )
        try:
            await self._authenticate_ws(ws)
        except Exception:
            await ws.close()
            raise
        self._ws = ws

    async def _authenticate_ws(self, ws: Any):
        """Authenticate via first-frame in-band auth when a WS secret is configured."""
        if not self.ws_secret:
            return
        await ws.send(json.dumps({"type": "auth", "token": self.ws_secret}))
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("ws auth timeout") from exc
        if not isinstance(raw, str):
            raise RuntimeError("ws auth invalid response type")
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("ws auth invalid JSON response") from exc
        mtype = msg.get("type")
        if mtype == "auth_ok":
            return
        if mtype == "error":
            raise RuntimeError(f"ws auth failed: {msg.get('text', 'unknown')}")
        raise RuntimeError(f"ws auth unexpected response: {mtype!r}")

    async def _close_ws(self):
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def close(self):
        """Shut down gracefully: finalize session, cancel push listener, close WS."""
        # Ask server to save session facts before disconnecting
        if self._ws is not None:
            finalize_targets = sorted(self._known_session_keys) or [None]
            for session_key in finalize_targets:
                payload = {"type": "finalize"}
                if session_key:
                    payload["session_key"] = session_key
                try:
                    await self.send_to_liagent(payload)
                except Exception:
                    pass
        if self._push_task and not self._push_task.done():
            self._push_task.cancel()
            try:
                await self._push_task
            except (asyncio.CancelledError, Exception):
                pass
            self._push_task = None
        await self._close_ws()
        await super().close()

    async def send_to_liagent(self, payload: dict) -> list[dict]:
        """Send a message to LiAgent and collect all response events until terminal.

        The lock is held for the entire send+receive cycle so that a
        concurrent caller cannot interleave messages on the same socket.
        """
        self._remember_session_key(payload.get("session_key"))
        async with self._response_lock:
            async with self._ws_lock:
                await self._ensure_ws()
                await self._ws.send(json.dumps(payload))

                events: list[dict] = []
                while True:
                    try:
                        raw = await asyncio.wait_for(self._ws.recv(), timeout=120)
                    except asyncio.TimeoutError:
                        events.append({"type": "error", "text": "response timeout"})
                        break
                    except (ConnectionClosed, ConnectionError, OSError) as e:
                        _log.warning("ws_disconnected", error=str(e))
                        self._ws = None
                        break
                    try:
                        msg = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        _log.warning("ws_malformed_json", raw_head=str(raw)[:120])
                        continue
                    events.append(msg)
                    mtype = msg.get("type")
                    if mtype in ("done", "error", "tts_done", "cleared", "tool_confirm_result", "finalized"):
                        # For voice mode, wait for tts_done after done
                        if mtype == "done" and msg.get("voice_pending"):
                            continue
                        break
                return events

    # ── Event handlers ───────────────────────────────────────────────────

    async def on_ready(self):
        print(f"  Discord bot online: {self.user} ({self.user.id})")
        # Sync slash commands
        try:
            await self.sync_commands()
        except Exception:
            pass
        # Start push listener for autonomous task results
        if self._push_task is None or self._push_task.done():
            self._push_task = asyncio.create_task(self._push_listener())

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if not self._should_respond(message):
            return

        # Check for voice message attachment (OGG/Opus)
        voice_attachment = None
        for att in message.attachments:
            if _is_audio_attachment(att.content_type, att.filename):
                voice_attachment = att
                break

        if voice_attachment:
            await self._handle_voice_message(
                message,
                voice_attachment,
                session_key=self._session_key_for_message(message),
            )
            return

        image_attachment = None
        for att in message.attachments:
            if _is_image_attachment(att.content_type, att.filename):
                image_attachment = att
                break

        text = self._extract_text(message)
        image_data_url: str | None = None
        vision_note: str | None = None
        if image_attachment is not None:
            try:
                image_bytes = await image_attachment.read()
                if image_bytes:
                    image_data_url, vision_note, cache_update = _prepare_discord_image(
                        image_bytes,
                        content_type=image_attachment.content_type,
                        filename=image_attachment.filename,
                        vision_cache=self._vision_cache,
                    )
                    if cache_update:
                        # Clean up old cache file
                        if self._vision_cache:
                            from .vision_routes import _cleanup_path
                            old_path = str(self._vision_cache.get("path", ""))
                            if old_path and old_path != str(cache_update.get("path", "")):
                                _cleanup_path(old_path)
                        self._vision_cache = cache_update
                    _log.event(
                        "discord_image",
                        filename=image_attachment.filename,
                        raw_bytes=len(image_bytes),
                        accepted=image_data_url is not None,
                        note=vision_note,
                    )
            except Exception as e:
                await message.reply(f"Image attachment error: {e}")
                return
            # Notify user if image was rejected
            if vision_note and image_data_url is None:
                await message.reply(f"Image skipped: {vision_note}")
                if not text:
                    return

        if not text and not image_data_url:
            return

        if not text and image_data_url:
            text = (
                str(
                    os.environ.get(
                        "LIAGENT_DISCORD_IMAGE_PROMPT",
                        "Please analyze this image.",
                    )
                    or ""
                ).strip()
                or "Please analyze this image."
            )

        await self._handle_text_message(
            message,
            text,
            image_data_url=image_data_url,
            session_key=self._session_key_for_message(message),
        )

    async def _handle_text_message(
        self,
        message: discord.Message,
        text: str,
        *,
        image_data_url: str | None = None,
        session_key: str | None = None,
    ):
        """Handle a text or text+image message — send to LiAgent and reply."""
        # Auto-set push channel so task results can be delivered
        if not self._push_channel_id:
            self._push_channel_id = message.channel.id
        async with message.channel.typing():
            try:
                payload = {"type": "text", "text": text}
                if image_data_url:
                    payload["image"] = image_data_url
                if session_key:
                    payload["session_key"] = session_key
                events = await self.send_to_liagent(payload)
            except Exception as e:
                await message.reply(f"Connection error: {e}")
                self._ws = None
                return

        await self._send_reply(message, events)

    async def _handle_voice_message(
        self,
        message: discord.Message,
        attachment: discord.Attachment,
        *,
        session_key: str | None = None,
    ):
        """Handle a voice message attachment — download, transcode, STT, respond."""
        async with message.channel.typing():
            try:
                # Download the audio file
                ogg_bytes = await attachment.read()
                print(f"  [voice-msg] downloaded {attachment.filename} ({len(ogg_bytes)} bytes)")

                # Convert to 16kHz mono float32
                loop = asyncio.get_event_loop()
                f32_bytes = await loop.run_in_executor(
                    None, _ogg_to_16k_mono_f32, ogg_bytes
                )
                audio_b64 = base64.b64encode(f32_bytes).decode()
                print(f"  [voice-msg] converted to f32 ({len(f32_bytes)} bytes)")

                # Send to LiAgent as audio
                events = await self.send_to_liagent(
                    {
                        "type": "audio",
                        "audio": audio_b64,
                        "session_key": session_key,
                    }
                )
            except Exception as e:
                await message.reply(f"Voice message error: {e}")
                return

        # Extract STT text, final response, and TTS audio
        stt_text = ""
        final_text = ""
        error_text = ""
        tts_chunks: list[tuple[np.ndarray, int]] = []
        for ev in events:
            mtype = ev.get("type", "")
            if mtype == "stt_result":
                stt_text = ev.get("text", "")
            elif mtype == "tts_chunk":
                audio_data = base64.b64decode(ev["audio"])
                audio_np = np.frombuffer(audio_data, dtype=np.float32)
                sr = ev.get("sample_rate", 24000)
                tts_chunks.append((audio_np, sr))
            elif mtype == "done":
                final_text = ev.get("text", "")
            elif mtype == "error":
                error_text = ev.get("text", "")

        if stt_text:
            print(f"  [voice-msg] STT: {stt_text}")

        if error_text and not final_text:
            await message.reply(f"Error: {error_text}")
            return

        # Build text reply
        parts = []
        if stt_text:
            parts.append(f"> {stt_text}")
        parts.append(final_text or "(no response)")
        reply_text = "\n\n".join(parts)

        # If we got TTS audio, send as voice message attachment
        if tts_chunks:
            combined = np.concatenate([c[0] for c in tts_chunks])
            sr = tts_chunks[0][1]
            print(f"  [voice-msg] encoding TTS reply ({combined.size} samples @ {sr}Hz)")
            loop = asyncio.get_event_loop()
            ogg_bytes = await loop.run_in_executor(
                None, _f32_to_ogg_opus, combined.tobytes(), sr
            )
            audio_file = discord.File(
                io.BytesIO(ogg_bytes), filename="reply.ogg"
            )
            # Send text + audio together
            first_chunk = reply_text[:2000]
            await message.reply(first_chunk, file=audio_file)
            for i in range(2000, len(reply_text), 2000):
                await message.channel.send(reply_text[i : i + 2000])
        else:
            # Text-only fallback
            chunks = [reply_text[i : i + 2000] for i in range(0, len(reply_text), 2000)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await message.reply(chunk)
                else:
                    await message.channel.send(chunk)

    async def _send_reply(self, message: discord.Message, events: list[dict]):
        """Extract response from events and send as Discord reply.

        Handles confirmation_required events by showing Approve/Reject buttons,
        then collecting the tool execution result and appending it to the reply.
        """
        final_text = ""
        error_text = ""
        vision_note = ""
        confirm_event = None
        suggestion_events = []
        for ev in events:
            if ev.get("type") == "done":
                final_text = ev.get("text", "")
            elif ev.get("type") == "error":
                error_text = ev.get("text", "")
            elif ev.get("type") == "vision_note":
                vision_note = ev.get("text", "")
            elif ev.get("type") == "confirmation_required":
                confirm_event = ev
            elif ev.get("type") == "proactive_suggestion":
                suggestion_events.append(ev)

        if vision_note:
            await message.channel.send(f"*Vision: {vision_note}*")

        # ── Confirmation flow ────────────────────────────────────────
        if confirm_event is not None:
            token = confirm_event.get("token", "")
            tool = confirm_event.get("tool", "unknown")
            reason = confirm_event.get("reason", "")
            brief = confirm_event.get("brief", "")
            view = _ConfirmationView(
                self,
                token,
                tool,
                brief,
                session_key=self._session_key_for_message(message),
            )
            confirm_text = f"**Tool confirmation required:** `{tool}`\n{reason}"
            await message.reply(confirm_text, view=view)
            result = await view.wait_for_result()
            # Send the outcome
            if result and result.get("status") == "ok":
                answer = result.get("answer", "")
                if answer:
                    chunks = [answer[i : i + 2000] for i in range(0, len(answer), 2000)]
                    for chunk in chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(f"`{tool}` executed successfully.")
            elif result and result.get("status") == "rejected":
                await message.channel.send(f"`{tool}` rejected.")
            elif result and result.get("status") == "timeout":
                await message.channel.send(f"Confirmation for `{tool}` timed out.")
            elif result and result.get("status") == "error":
                await message.channel.send(f"Error: {result.get('message', 'unknown')}")
            return

        # ── Proactive suggestion flow (non-blocking, shown alongside reply) ──
        for sev in suggestion_events:
            sug_msg = sev.get("message", "")
            sug_id = sev.get("suggestion_id", "")
            if sug_msg:
                view = _ProactiveSuggestionView(
                    self,
                    sug_id,
                    session_key=self._session_key_for_message(message),
                )
                await message.channel.send(f"**Suggestion:** {sug_msg}", view=view)

        # ── Normal reply flow ────────────────────────────────────────
        if error_text and not final_text:
            await message.reply(f"Error: {error_text}")
            return

        if not final_text:
            await message.reply("(no response)")
            return

        chunks = [final_text[i : i + 2000] for i in range(0, len(final_text), 2000)]
        for i, chunk in enumerate(chunks):
            if i == 0:
                await message.reply(chunk)
            else:
                await message.channel.send(chunk)

    def _should_respond(self, message: discord.Message) -> bool:
        """Respond to DMs or @mentions in servers."""
        # Always respond to DMs
        if isinstance(message.channel, discord.DMChannel):
            return True
        # Respond to @mentions in servers
        if self.user and self.user.mentioned_in(message):
            return True
        return False

    def _extract_text(self, message: discord.Message) -> str:
        """Extract clean text from message, removing @mention."""
        text = message.content
        if self.user:
            text = text.replace(f"<@{self.user.id}>", "").replace(
                f"<@!{self.user.id}>", ""
            )
        return text.strip()

    # ── Push listener ──────────────────────────────────────────────────

    async def _push_listener(self):
        """Background task: connect to /ws/task-push and forward results to Discord."""
        push_url = self.ws_url.replace("/ws/chat", "/ws/task-push")
        while True:
            ws = None
            try:
                async with websockets.connect(push_url, ping_interval=30) as ws:
                    await self._authenticate_ws(ws)
                    self._push_ws = ws
                    await self._sync_push_session_subscriptions()
                    print("  [push] connected to task-push")
                    while True:
                        raw = await ws.recv()
                        msg = json.loads(raw)
                        await self._dispatch_push_message(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._push_ws is not None:
                    self._push_ws = None
                print(f"  [push] connection error: {e}, reconnecting in 10s...")
                await asyncio.sleep(10)
            finally:
                if ws is not None and self._push_ws is ws:
                    self._push_ws = None

    async def _dispatch_push_message(self, msg: dict):
        """Route a task-push event to the correct Discord delivery path."""
        msg_type = msg.get("type")
        if msg_type == "task_result":
            await self._deliver_push(msg)
        elif msg_type == "heartbeat_confirm":
            await self._deliver_heartbeat_confirm(msg)
        elif msg_type == "proactive_suggestion":
            await self._deliver_proactive_suggestion(msg)

    async def _deliver_push(self, result: dict):
        """Send a task result to the configured Discord push channel."""
        if not self._push_channel_id:
            return
        channel = self.get_channel(self._push_channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(self._push_channel_id)
            except Exception:
                return

        task_name = result.get("task_name", "Unknown")
        status = result.get("status", "")
        text = result.get("result", "") or result.get("error", "")

        embed = discord.Embed(
            title=f"Task: {task_name}",
            color=discord.Color.green() if status == "success" else discord.Color.red(),
        )
        embed.add_field(name="Status", value=status, inline=True)
        embed.add_field(name="Run ID", value=result.get("run_id", ""), inline=True)
        if text:
            # Truncate to Discord embed limit
            embed.description = text[:4000]
        try:
            await channel.send(embed=embed)
        except Exception as e:
            print(f"  [push] failed to send to channel: {e}")

    async def _deliver_heartbeat_confirm(self, msg: dict):
        """Send a heartbeat confirmation card to the push channel."""
        if not self._push_channel_id:
            return
        channel = self.get_channel(self._push_channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(self._push_channel_id)
            except Exception:
                return

        description = msg.get("description", "Unknown action")
        embed = discord.Embed(
            title="Heartbeat: Confirmation Required",
            description=description,
            color=discord.Color.orange(),
        )
        embed.add_field(name="Tool", value=msg.get("action_type", "?"), inline=True)
        embed.add_field(name="Risk", value=msg.get("risk_level", "?"), inline=True)
        embed.add_field(name="Expires", value=msg.get("expires_at", "?"), inline=True)
        action_args = msg.get("action_args")
        if action_args:
            import json as _json
            embed.add_field(
                name="Arguments",
                value=_json.dumps(action_args, ensure_ascii=False)[:1024],
                inline=False,
            )

        # Derive API base from ws_url
        api_base = self.ws_url.replace("/ws/chat", "").replace("ws://", "http://").replace("wss://", "https://")

        view = _HeartbeatConfirmView(
            self, msg["token"], description, msg.get("expires_at", ""),
            api_base=api_base, api_secret=self.ws_secret or "",
        )
        try:
            await channel.send(embed=embed, view=view)
        except Exception as e:
            print(f"  [push] failed to send heartbeat confirm: {e}")

    async def _deliver_proactive_suggestion(self, msg: dict):
        """Send a proactive suggestion card to the push channel."""
        target_session_id = str(msg.get("target_session_id", "") or "").strip()
        channel_id = self._push_channel_id
        if target_session_id:
            channel_id = self._session_channels.get(target_session_id)
        if not channel_id:
            return
        channel = self.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(channel_id)
            except Exception:
                return

        suggestion_id = msg.get("suggestion_id", "")
        embed = discord.Embed(
            title="Suggestion",
            description=msg.get("message", "") or "New proactive suggestion",
            color=discord.Color.blurple(),
        )
        if msg.get("domain"):
            embed.add_field(name="Domain", value=msg["domain"], inline=True)
        if msg.get("suggestion_type"):
            embed.add_field(name="Type", value=msg["suggestion_type"], inline=True)
        view = _ProactiveSuggestionView(
            self,
            suggestion_id,
            session_key=msg.get("target_session_id"),
        )
        try:
            await channel.send(embed=embed, view=view)
        except Exception as e:
            print(f"  [push] failed to send proactive suggestion: {e}")

    # ── Slash commands ───────────────────────────────────────────────────

    async def _cmd_join(self, ctx: discord.ApplicationContext):
        """Join the user's voice channel."""
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.respond("You're not in a voice channel.", ephemeral=True)
            return

        await ctx.defer()
        vc = await ctx.author.voice.channel.connect()
        loop = asyncio.get_running_loop()
        sink = LiAgentVoiceSink(self, ctx.channel, loop)
        vc.start_recording(sink, self._on_recording_done, ctx.channel)
        await ctx.followup.send(f"Joined **{ctx.author.voice.channel.name}**. Listening...")

    async def _on_recording_done(self, sink: LiAgentVoiceSink, channel):
        """Called when recording stops."""
        sink.cleanup()

    async def _cmd_leave(self, ctx: discord.ApplicationContext):
        """Leave the voice channel."""
        if not ctx.voice_client:
            await ctx.respond("Not in a voice channel.", ephemeral=True)
            return
        await ctx.defer()
        await ctx.voice_client.disconnect()
        await ctx.followup.send("Left voice channel.")

    async def _cmd_clear(self, ctx: discord.ApplicationContext):
        """Clear LiAgent conversation memory."""
        await ctx.defer()
        try:
            _ = await self.send_to_liagent(
                {"type": "clear", "session_key": self._session_key_for_ctx(ctx)}
            )
            await ctx.followup.send("Conversation memory cleared.")
        except Exception as e:
            try:
                await ctx.followup.send(f"Error: {e}")
            except Exception:
                _log.error("cmd_clear_followup_failed", error=str(e))

    def _api_base(self) -> str:
        return _get_api_base(self.ws_url)

    def _api_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.ws_secret:
            headers["x-liagent-token"] = self.ws_secret
        return headers

    async def _cmd_speaker(
        self, ctx: discord.ApplicationContext, name: str
    ):
        """Change TTS speaker voice."""
        import httpx

        await ctx.defer()
        api_base = self._api_base()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{api_base}/api/config/tts_voice",
                    json={"speaker_name": name},
                    headers=self._api_headers(),
                    timeout=10,
                )
                data = resp.json()
                if resp.status_code == 200:
                    await ctx.followup.send(f"TTS speaker changed to **{name}**.")
                else:
                    await ctx.followup.send(
                        f"Error: {data.get('error', 'unknown')}"
                    )
        except Exception as e:
            try:
                await ctx.followup.send(f"Error: {e}")
            except Exception:
                _log.error("cmd_speaker_followup_failed", error=str(e))

    async def _cmd_status(self, ctx: discord.ApplicationContext):
        """Show LiAgent engine status."""
        import httpx

        await ctx.defer()
        api_base = self._api_base()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{api_base}/api/health",
                    headers=self._api_headers(),
                    timeout=10,
                )
                health = resp.json()

            embed = discord.Embed(
                title="LiAgent Status",
                color=discord.Color.green()
                if health.get("status") == "ok"
                else discord.Color.red(),
            )
            for key, value in health.items():
                if isinstance(value, dict):
                    value = json.dumps(value, indent=2)[:1024]
                embed.add_field(
                    name=key, value=f"```{value}```", inline=False
                )
            await ctx.followup.send(embed=embed)
        except Exception as e:
            try:
                await ctx.followup.send(f"Error: {e}")
            except Exception:
                _log.error("cmd_status_followup_failed", error=str(e))

    async def _cmd_repl_mode(
        self,
        ctx: discord.ApplicationContext,
        mode: str,
        *,
        confirm_trusted_local: bool = False,
    ):
        """Set REPL safety mode."""
        import httpx

        selected = str(mode or "").strip().lower()
        if selected not in {"off", "sandboxed", "trusted_local"}:
            await ctx.respond("Invalid mode. Use off|sandboxed|trusted_local.", ephemeral=True)
            return

        await ctx.defer(ephemeral=True)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._api_base()}/api/config/repl_mode",
                    json={
                        "repl_mode": selected,
                        "confirm_trusted_local": bool(confirm_trusted_local),
                    },
                    headers=self._api_headers(),
                    timeout=15,
                )
                data = resp.json()
            if resp.status_code >= 400:
                await ctx.followup.send(
                    f"Error ({resp.status_code}): {data.get('error', 'unknown')}",
                    ephemeral=True,
                )
                return
            mode_now = str(data.get("repl_mode", selected))
            if mode_now == "trusted_local":
                msg = "REPL mode set to `trusted_local` (unsafe for untrusted code)."
            elif mode_now == "off":
                msg = "REPL mode set to `off`."
            else:
                msg = "REPL mode set to `sandboxed`."
            await ctx.followup.send(msg, ephemeral=True)
        except Exception as e:
            await ctx.followup.send(f"Error: {e}", ephemeral=True)

    async def _cmd_repl_status(self, ctx: discord.ApplicationContext, scope: str = "current"):
        """Show REPL status for current session or all sessions."""
        import httpx

        scope_value = str(scope or "current").strip().lower()
        session_id = None if scope_value == "all" else self._session_key_for_ctx(ctx)
        params = {"session_id": session_id} if session_id else {}

        await ctx.defer(ephemeral=True)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self._api_base()}/api/repl/status",
                    params=params,
                    headers=self._api_headers(),
                    timeout=15,
                )
                data = resp.json()
            if resp.status_code >= 400:
                await ctx.followup.send(
                    f"Error ({resp.status_code}): {data.get('error', 'unknown')}",
                    ephemeral=True,
                )
                return
            repl = data.get("repl", {}) if isinstance(data, dict) else {}
            mode = repl.get("mode", "unknown")
            if scope_value == "all":
                sessions = repl.get("sessions", [])
                total = repl.get("total_sessions", 0)
                lines = [f"mode={mode}", f"total_sessions={total}"]
                for s in sessions[:10]:
                    lines.append(
                        f"{s.get('session_id')} alive={s.get('alive')} idle={s.get('idle_seconds')}s"
                    )
            else:
                lines = [
                    f"mode={mode}",
                    f"session_id={repl.get('session_id', session_id)}",
                    f"exists={repl.get('exists')}",
                ]
                if repl.get("exists"):
                    lines.append(f"alive={repl.get('alive')}")
                    lines.append(f"idle_seconds={repl.get('idle_seconds')}")
            await ctx.followup.send("```text\n" + "\n".join(lines) + "\n```", ephemeral=True)
        except Exception as e:
            await ctx.followup.send(f"Error: {e}", ephemeral=True)

    async def _cmd_repl_reset(self, ctx: discord.ApplicationContext):
        """Reset current Discord scoped REPL session."""
        import httpx

        session_id = self._session_key_for_ctx(ctx)
        await ctx.defer(ephemeral=True)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._api_base()}/api/repl/reset",
                    json={"session_id": session_id},
                    headers=self._api_headers(),
                    timeout=15,
                )
                data = resp.json()
            if resp.status_code >= 400:
                await ctx.followup.send(
                    f"Error ({resp.status_code}): {data.get('error', 'unknown')}",
                    ephemeral=True,
                )
                return
            await ctx.followup.send("Current REPL session has been reset.", ephemeral=True)
        except Exception as e:
            await ctx.followup.send(f"Error: {e}", ephemeral=True)

    async def _cmd_repl_kill(self, ctx: discord.ApplicationContext):
        """Kill current Discord scoped REPL worker session."""
        import httpx

        session_id = self._session_key_for_ctx(ctx)
        await ctx.defer(ephemeral=True)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._api_base()}/api/repl/kill",
                    json={"session_id": session_id},
                    headers=self._api_headers(),
                    timeout=15,
                )
                data = resp.json()
            if resp.status_code >= 400:
                await ctx.followup.send(
                    f"Error ({resp.status_code}): {data.get('error', 'unknown')}",
                    ephemeral=True,
                )
                return
            await ctx.followup.send("Current REPL session has been killed.", ephemeral=True)
        except Exception as e:
            await ctx.followup.send(f"Error: {e}", ephemeral=True)

    def _register_slash_commands(self):
        """Register slash commands with the bot."""
        self.slash_command(name="join", description="Join your voice channel")(
            self._cmd_join
        )
        self.slash_command(name="leave", description="Leave voice channel")(
            self._cmd_leave
        )
        self.slash_command(name="clear", description="Clear conversation memory")(
            self._cmd_clear
        )

        # /speaker with name parameter
        speaker_cmd = self.slash_command(
            name="speaker", description="Change TTS speaker voice"
        )

        @speaker_cmd
        async def speaker_wrapper(
            ctx: discord.ApplicationContext,
            name: discord.Option(
                str,
                description="Speaker name",
                choices=[
                    "serena", "vivian", "uncle_fu", "ryan",
                    "aiden", "ono_anna", "sohee", "eric", "dylan",
                ],
            ),
        ):
            await self._cmd_speaker(ctx, name)

        self.slash_command(name="status", description="Show LiAgent engine status")(
            self._cmd_status
        )

        repl_mode_cmd = self.slash_command(
            name="repl-mode",
            description="Set REPL mode (off/sandboxed/trusted_local)",
        )

        @repl_mode_cmd
        async def repl_mode_wrapper(
            ctx: discord.ApplicationContext,
            mode: discord.Option(
                str,
                description="REPL mode",
                choices=["off", "sandboxed", "trusted_local"],
            ),
            confirm: discord.Option(
                str,
                description="Must be yes for trusted_local",
                choices=["no", "yes"],
                default="no",
            ),
        ):
            await self._cmd_repl_mode(
                ctx,
                str(mode),
                confirm_trusted_local=(str(confirm).lower() == "yes"),
            )

        repl_status_cmd = self.slash_command(
            name="repl-status",
            description="Show REPL mode/session status",
        )

        @repl_status_cmd
        async def repl_status_wrapper(
            ctx: discord.ApplicationContext,
            scope: discord.Option(
                str,
                description="current or all sessions",
                choices=["current", "all"],
                default="current",
            ),
        ):
            await self._cmd_repl_status(ctx, str(scope))

        self.slash_command(
            name="repl-reset",
            description="Reset current Discord scoped REPL session",
        )(self._cmd_repl_reset)
        self.slash_command(
            name="repl-kill",
            description="Kill current Discord scoped REPL session",
        )(self._cmd_repl_kill)

        # ── /task command group ──────────────────────────────────────────
        task_group = self.create_group("task", "Manage autonomous tasks")

        @task_group.command(name="create", description="Create an autonomous task")
        async def task_create(
            ctx: discord.ApplicationContext,
            description: discord.Option(str, description="Task description in natural language"),
        ):
            # Auto-set push channel so results are delivered here
            if not self._push_channel_id:
                self._push_channel_id = ctx.channel_id
            await self._task_api(ctx, "POST", "/api/tasks", json={"text": description})

        @task_group.command(name="list", description="List all autonomous tasks")
        async def task_list(ctx: discord.ApplicationContext):
            await self._task_api(ctx, "GET", "/api/tasks")

        @task_group.command(name="pause", description="Pause an autonomous task")
        async def task_pause(
            ctx: discord.ApplicationContext,
            task_id: discord.Option(str, description="Task ID to pause"),
        ):
            await self._task_api(ctx, "POST", f"/api/tasks/{task_id}/pause")

        @task_group.command(name="resume", description="Resume a paused task")
        async def task_resume(
            ctx: discord.ApplicationContext,
            task_id: discord.Option(str, description="Task ID to resume"),
        ):
            await self._task_api(ctx, "POST", f"/api/tasks/{task_id}/resume")

        @task_group.command(name="delete", description="Delete an autonomous task")
        async def task_delete(
            ctx: discord.ApplicationContext,
            task_id: discord.Option(str, description="Task ID to delete"),
        ):
            await self._task_api(ctx, "DELETE", f"/api/tasks/{task_id}")

        # /task-channel command to set push channel
        @self.slash_command(name="task-channel", description="Set this channel for task result pushes")
        async def set_task_channel(ctx: discord.ApplicationContext):
            self._push_channel_id = ctx.channel_id
            await ctx.respond(f"Task results will be pushed to this channel.")

        # ── /watch command group ─────────────────────────────────────────
        watch_group = self.create_group("watch", "Proactive monitoring")

        @watch_group.command(name="create", description="Start monitoring a topic")
        async def watch_create(
            ctx: discord.ApplicationContext,
            query: discord.Option(str, description="What to monitor, e.g. 'Watch AAPL, cost basis 142'"),
        ):
            await self._watch_create(ctx, query)

        @watch_group.command(name="list", description="List active monitors")
        async def watch_list(ctx: discord.ApplicationContext):
            await self._watch_api(ctx, "GET", "/api/interests")

        @watch_group.command(name="pause", description="Pause a monitor")
        async def watch_pause(
            ctx: discord.ApplicationContext,
            interest_id: discord.Option(str, description="Interest ID to pause"),
        ):
            await self._watch_api(ctx, "POST", f"/api/interests/{interest_id}/pause")

        @watch_group.command(name="resume", description="Resume a paused monitor")
        async def watch_resume(
            ctx: discord.ApplicationContext,
            interest_id: discord.Option(str, description="Interest ID to resume"),
        ):
            await self._watch_api(ctx, "POST", f"/api/interests/{interest_id}/resume")

        @watch_group.command(name="delete", description="Stop and archive a monitor")
        async def watch_delete(
            ctx: discord.ApplicationContext,
            interest_id: discord.Option(str, description="Interest ID to delete"),
        ):
            await self._watch_api(ctx, "DELETE", f"/api/interests/{interest_id}")

    # ── Watch helpers ────────────────────────────────────────────────────

    async def _watch_create(self, ctx: discord.ApplicationContext, query: str):
        """Create an interest, open a Discord thread, and post the coverage embed."""
        import httpx

        api_base = _get_api_base(self.ws_url)
        headers = {}
        if self.ws_secret:
            headers["x-liagent-token"] = self.ws_secret

        await ctx.defer()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{api_base}/api/interests",
                    json={"query": query},
                    headers=headers,
                    timeout=60,  # LLM factor generation may take time
                )
                data = resp.json()

            if resp.status_code >= 400:
                await ctx.respond(
                    f"Error ({resp.status_code}): {data.get('error', 'unknown')}",
                    ephemeral=True,
                )
                return

            coverage = data.get("coverage", {})
            interest = data.get("interest", {})
            interest_id = interest.get("id", "")
            intent = coverage.get("intent", query)

            # Respond first, then create thread from the response message
            reply_msg = await ctx.respond(
                f"Monitor `{interest_id}` created — opening thread..."
            )
            # fetch the actual Message object from the interaction
            msg = await ctx.interaction.original_response()

            # Create thread attached to the response message
            thread_name = f"Watch: {intent[:90]}"
            thread = await msg.create_thread(name=thread_name)

            # Update backend with the thread ID
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{api_base}/api/interests/{interest_id}/thread",
                        json={"discord_thread_id": str(thread.id)},
                        headers=headers,
                        timeout=10,
                    )
            except Exception:
                pass  # non-critical — embed still works

            # Post coverage embed as first message in thread
            embed = self._build_coverage_embed(coverage, interest_id)
            await thread.send(embed=embed)

        except Exception as e:
            await ctx.respond(f"Error: {e}", ephemeral=True)

    async def _watch_api(
        self,
        ctx: discord.ApplicationContext,
        method: str,
        path: str,
    ):
        """Generic helper for watch CRUD operations (list/pause/resume/delete)."""
        import httpx

        api_base = _get_api_base(self.ws_url)
        headers = {}
        if self.ws_secret:
            headers["x-liagent-token"] = self.ws_secret

        await ctx.defer()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    method, f"{api_base}{path}",
                    headers=headers, timeout=30,
                )
                data = resp.json()

            if resp.status_code >= 400:
                await ctx.respond(
                    f"Error ({resp.status_code}): {data.get('error', 'unknown')}",
                    ephemeral=True,
                )
                return

            # Format response based on content
            if "interests" in data:
                interests = data["interests"]
                if not interests:
                    await ctx.respond("No active monitors.")
                    return
                lines = []
                for it in interests[:15]:
                    icon = {"active": "\u25b6", "paused": "\u23f8"}.get(
                        it.get("status", ""), "?"
                    )
                    ratio = it.get("coverage_ratio", 0)
                    pct = f"{ratio * 100:.0f}%"
                    lines.append(
                        f"`{it['id']}` {icon} **{it.get('intent') or it.get('query', '')}** "
                        f"({pct} coverage)"
                    )
                await ctx.respond("\n".join(lines))
            else:
                status = data.get("status", "ok")
                await ctx.respond(f"Done: {status}")
        except Exception as e:
            await ctx.respond(f"Error: {e}", ephemeral=True)

    @staticmethod
    def _build_coverage_embed(coverage: dict, interest_id: str) -> "discord.Embed":
        """Build a Discord Embed showing factor coverage breakdown."""
        intent = coverage.get("intent", "")
        ctx_dict = coverage.get("context", {})
        ratio = coverage.get("coverage_ratio", 0)
        total = coverage.get("total", 0)
        active = coverage.get("active", 0)

        pct = f"{ratio * 100:.0f}%"
        color = (
            discord.Color.green() if ratio >= 0.8
            else discord.Color.gold() if ratio >= 0.5
            else discord.Color.red()
        )

        embed = discord.Embed(
            title=f"Monitor: {intent}",
            color=color,
        )

        # Context line
        if ctx_dict:
            ctx_parts = [f"{k}: {v}" for k, v in ctx_dict.items()]
            embed.description = " | ".join(ctx_parts)

        # Executable factors
        exe_list = coverage.get("executable", [])
        if exe_list:
            names = [f"{f['name']}" + (f" ({f['entity']})" if f.get("entity") else "")
                     for f in exe_list]
            embed.add_field(
                name=f"\u2705 Executable ({len(exe_list)})",
                value="\n".join(names)[:1024],
                inline=False,
            )

        # Proxy factors
        proxy_list = coverage.get("proxy", [])
        if proxy_list:
            names = [f"{f['name']}" + (f" ({f['entity']})" if f.get("entity") else "")
                     for f in proxy_list]
            embed.add_field(
                name=f"\U0001f50d Proxy ({len(proxy_list)})",
                value="\n".join(names)[:1024],
                inline=False,
            )

        # Blind factors
        blind_list = coverage.get("blind", [])
        if blind_list:
            names = [f"{f['name']} [{f.get('source_hint', '')}]" for f in blind_list]
            embed.add_field(
                name=f"\U0001f6ab Blind ({len(blind_list)})",
                value="\n".join(names)[:1024],
                inline=False,
            )

        embed.set_footer(text=f"Coverage: {pct} ({active}/{total}) | ID: {interest_id}")
        return embed

    async def _task_api(
        self,
        ctx: discord.ApplicationContext,
        method: str,
        path: str,
        *,
        json: dict | None = None,
    ):
        """Helper: call the LiAgent REST API and respond with the result."""
        import httpx

        api_base = _get_api_base(self.ws_url)
        headers = {}
        if self.ws_secret:
            headers["x-liagent-token"] = self.ws_secret

        await ctx.defer()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    method, f"{api_base}{path}",
                    json=json, headers=headers, timeout=30,
                )
                data = resp.json()

            if resp.status_code >= 400:
                await ctx.respond(
                    f"Error ({resp.status_code}): {data.get('error', 'unknown')}",
                    ephemeral=True,
                )
                return

            # Format response
            if "tasks" in data:
                tasks = data["tasks"]
                if not tasks:
                    await ctx.respond("No tasks found.")
                    return
                lines = []
                for t in tasks[:15]:
                    status_icon = {"active": "▶", "paused": "⏸", "deleted": "🗑"}.get(t.get("status", ""), "?")
                    lines.append(
                        f"`{t['id']}` {status_icon} **{t['name']}** ({t.get('trigger_type', '')})"
                    )
                await ctx.respond("\n".join(lines))
            elif "task" in data:
                t = data["task"]
                await ctx.respond(f"Created task `{t['id']}`: **{t['name']}** ({t.get('trigger_type', '')})")
            else:
                status = data.get("status", "ok")
                await ctx.respond(f"Done: {status}")
        except Exception as e:
            await ctx.respond(f"Error: {e}", ephemeral=True)


# ─── Factory ────────────────────────────────────────────────────────────────

_bot_instance: LiAgentBot | None = None


def get_bot_instance() -> LiAgentBot | None:
    """Return the active bot instance (used by signal_poller callback)."""
    return _bot_instance


def create_discord_bot(ws_url: str, ws_secret: str) -> LiAgentBot:
    """Create and configure the Discord bot instance."""
    global _bot_instance
    bot = LiAgentBot(ws_url=ws_url, ws_secret=ws_secret)
    bot._register_slash_commands()
    _bot_instance = bot
    return bot
