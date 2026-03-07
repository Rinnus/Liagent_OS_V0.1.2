/**
 * Voice overlay renderer — mic capture, TTS playback, wave visualisation.
 * C1: Reads stores via getters, writes to DOM only.
 * Never mutates another domain's store directly.
 */
import {
  getVoiceMode, setVoiceMode,
  getVoiceState, setVoiceState as _setVoiceStateRaw,
  getIsRecording, setIsRecording,
  getAudioContext, setAudioContext,
  getTtsQueue, getTtsPlaying, setTtsPlaying,
  getTtsAudioCtx, setTtsAudioCtx,
  getTtsDone, setTtsDone,
  getTtsCurrentSource, setTtsCurrentSource,
  getLastVoiceImageUrl, setLastVoiceImageUrl,
  resetTTSState,
  mergeFloat32Arrays, float32ToBase64,
} from '../stores/voice-store.js';
import { getWebSessionKey } from '../stores/chat-store.js';

import { getWs, wsSend } from '../ws-send.js';

import {
  getPendingImage, setPendingImage,
  getCameraMode,
  getCameraFrameDataUrl, getCameraFrameAtMs,
  getCameraCanvas, setCameraCanvas,
  setCameraFrameDataUrl, setCameraFrameAtMs,
} from '../stores/media-store.js';

// ─── Voice detection thresholds (single reference point) ────────────
var VOICE_CONFIG = {
  SILENCE_THRESHOLD: 0.015,
  SILENCE_DURATION: 700,
  MAX_UTTERANCE_MS: 12000,
  BARGE_IN_THRESHOLD: 0.04,
  BARGE_IN_DURATION: 200,
};

// ─── Module-scoped voice/mic state (was window._voice* globals) ─────
var _voiceChunks = [];
var _voiceSpeaking = false;
var _voiceSilenceStart = 0;
var _voiceSpeechStart = 0;
var _voicePaused = false;
var _voiceProcessor = null;
var _voiceStream = null;
var _voiceSource = null;
var _bargeInSpeechStart = 0;

function resetVoiceState() {
  _voiceChunks = [];
  _voiceSpeaking = false;
  _voiceSilenceStart = 0;
  _voiceSpeechStart = 0;
  _voicePaused = false;
  _bargeInSpeechStart = 0;
}

// ─── Voice state (updates store + DOM) ─────────────────────────────

/**
 * Set voice state in store and update overlay DOM elements.
 */
export function setVoiceState(state) {
  _setVoiceStateRaw(state);
  var statusEl = document.getElementById('voice-status');
  var waveEl = document.getElementById('voice-wave');
  waveEl.classList.remove('active', 'responding');
  switch (state) {
    case 'listening':
      statusEl.textContent = 'listening';
      document.getElementById('voice-text').textContent = '';
      break;
    case 'hearing':
      statusEl.textContent = 'hearing...';
      waveEl.classList.add('active');
      break;
    case 'processing':
      statusEl.textContent = 'thinking...';
      break;
    case 'responding':
      statusEl.textContent = 'responding...';
      waveEl.classList.add('responding');
      break;
    case 'speaking':
      statusEl.textContent = 'speaking...';
      break;
  }
}

// ─── Voice lifecycle ───────────────────────────────────────────────

export function toggleVoice() {
  if (getVoiceMode()) stopVoice(); else startVoice();
}

export function startVoice() {
  resetVoiceState();
  setVoiceMode(true);
  unlockTTSAudio();
  document.getElementById('voice-overlay').classList.add('active');
  if (getCameraMode()) {
    document.getElementById('voice-cam-preview').classList.add('active');
  }
  document.getElementById('voice-text').textContent = '';
  initWaveBars();
  setVoiceState('listening');
  startMicCapture();
  // Show pending image in voice overlay if exists
  if (getPendingImage()) {
    // DOM update for voice overlay image preview
    var img = getPendingImage();
    document.getElementById('voice-img-thumb').src = img.dataUrl;
    document.getElementById('voice-img-preview').classList.add('active');
    document.getElementById('voice-img-btn').classList.add('has-image');
  }
}

export function stopVoice() {
  setVoiceMode(false);
  _setVoiceStateRaw('idle');
  stopMicCapture();
  document.getElementById('voice-overlay').classList.remove('active');
  document.getElementById('voice-cam-preview').classList.remove('active');
  document.getElementById('voice-wave').classList.remove('active', 'responding');
}

// ─── Mic pause / resume ────────────────────────────────────────────

export function pauseVoiceListening() {
  if (_voiceProcessor) {
    _voicePaused = true;
  }
}

export function resumeVoiceListening() {
  if (_voiceProcessor) {
    _voicePaused = false;
    _voiceChunks = [];
    _voiceSpeaking = false;
    _voiceSilenceStart = 0;
    _bargeInSpeechStart = 0;
  }
}

// ─── Barge-in ──────────────────────────────────────────────────────

export function bargeIn() {
  // Stop TTS playback and cancel current run
  var src = getTtsCurrentSource();
  if (src) {
    try { src.stop(); } catch(e) {}
    setTtsCurrentSource(null);
  }
  getTtsQueue().length = 0;   // C4: in-place clearing
  setTtsPlaying(false);
  setTtsDone(false);
  // Notify backend to cancel current generation
  wsSend({ type: 'barge_in' });
  // Reset voice state for new input
  _voiceChunks = [];
  _voiceSpeaking = false;
  _voiceSilenceStart = 0;
  _bargeInSpeechStart = 0;
  if (getVoiceMode()) {
    setVoiceState('listening');
  }
}

// ─── Camera frame capture (local helper) ───────────────────────────
// Duplicated from media-capture to avoid renderer-to-renderer import.
// Reads/writes media-store state only.

function _captureCameraFrame() {
  if (!getCameraMode()) return;
  var video = document.getElementById('cam-video');
  if (!video || !video.videoWidth || !video.videoHeight) return;
  var canvas = getCameraCanvas();
  if (!canvas) {
    canvas = document.createElement('canvas');
    setCameraCanvas(canvas);
  }
  var targetW = 640;
  var targetH = Math.max(240, Math.round(video.videoHeight * targetW / video.videoWidth));
  canvas.width = targetW;
  canvas.height = targetH;
  var ctx = canvas.getContext('2d', { alpha: false });
  if (!ctx) return;
  ctx.drawImage(video, 0, 0, targetW, targetH);
  setCameraFrameDataUrl(canvas.toDataURL('image/jpeg', 0.70));
  setCameraFrameAtMs(Date.now());
}

// ─── Mic capture ───────────────────────────────────────────────────

export function startMicCapture() {
  navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
    var ctx = new AudioContext({ sampleRate: 16000 });
    setAudioContext(ctx);
    var source = ctx.createMediaStreamSource(stream);
    var processor = ctx.createScriptProcessor(4096, 1, 1);

    _voiceChunks = [];
    _voiceSpeaking = false;
    _voiceSilenceStart = 0;
    _voiceSpeechStart = 0;
    _voicePaused = false;
    var SILENCE_THRESHOLD = VOICE_CONFIG.SILENCE_THRESHOLD;
    var SILENCE_DURATION = VOICE_CONFIG.SILENCE_DURATION;
    var MAX_UTTERANCE_MS = VOICE_CONFIG.MAX_UTTERANCE_MS;

    function flushVoiceUtterance() {
      if (!_voiceChunks || _voiceChunks.length === 0) return;
      var allAudio = mergeFloat32Arrays(_voiceChunks);
      _voiceChunks = [];
      _voiceSilenceStart = 0;
      _voiceSpeechStart = 0;
      pauseVoiceListening();
      setVoiceState('processing');
      var payload = {
        type: 'audio',
        audio: float32ToBase64(allAudio),
        sample_rate: 16000,
        reuse_vision: !!getCameraMode(),
        session_key: getWebSessionKey(),
      };
      var imageUrl = null;
      var pending = getPendingImage();
      if (pending) {
        payload.image = pending.base64;
        imageUrl = pending.dataUrl;
        // Clear pending image state + DOM
        setPendingImage(null);
        document.getElementById('img-preview').classList.remove('active');
        document.getElementById('img-btn').classList.remove('has-image');
        document.getElementById('voice-img-preview').classList.remove('active');
        document.getElementById('voice-img-btn').classList.remove('has-image');
      } else if (getCameraMode()) {
        _captureCameraFrame();
        var frameUrl = getCameraFrameDataUrl();
        if (frameUrl) {
          payload.image = frameUrl;
          var frameAt = getCameraFrameAtMs();
          if (frameAt > 0) payload.image_age_ms = Math.max(0, Date.now() - frameAt);
          imageUrl = frameUrl;
        }
      }
      setLastVoiceImageUrl(imageUrl);
      wsSend(payload);
    }

    var BARGE_IN_THRESHOLD = VOICE_CONFIG.BARGE_IN_THRESHOLD;
    var BARGE_IN_DURATION = VOICE_CONFIG.BARGE_IN_DURATION;

    processor.onaudioprocess = function(e) {
      var data = e.inputBuffer.getChannelData(0);
      var rms = Math.sqrt(data.reduce(function(s, v) { return s + v * v; }, 0) / data.length);

      // Barge-in detection: if TTS is playing and we detect sustained speech, interrupt
      if (_voicePaused && getTtsPlaying()) {
        if (rms > BARGE_IN_THRESHOLD) {
          if (!_bargeInSpeechStart) _bargeInSpeechStart = Date.now();
          if (Date.now() - _bargeInSpeechStart > BARGE_IN_DURATION) {
            bargeIn();
            _voicePaused = false;
          }
        } else {
          _bargeInSpeechStart = 0;
        }
        return;
      }
      if (_voicePaused) return;

      updateWave(rms);

      if (rms > SILENCE_THRESHOLD) {
        _voiceSpeaking = true;
        if (!_voiceSpeechStart) _voiceSpeechStart = Date.now();
        _voiceSilenceStart = 0;
        _voiceChunks.push(new Float32Array(data));
        if (getVoiceState() === 'listening') setVoiceState('hearing');
        if (Date.now() - _voiceSpeechStart > MAX_UTTERANCE_MS) {
          _voiceSpeaking = false;
          flushVoiceUtterance();
        }
      } else if (_voiceSpeaking) {
        _voiceChunks.push(new Float32Array(data));
        if (!_voiceSilenceStart) _voiceSilenceStart = Date.now();
        if (Date.now() - _voiceSilenceStart > SILENCE_DURATION) {
          _voiceSpeaking = false;
          flushVoiceUtterance();
        }
      }
    };

    source.connect(processor);
    processor.connect(ctx.destination);
    setIsRecording(true);
    document.getElementById('mic-btn').classList.add('recording');

    _voiceStream = stream;
    _voiceProcessor = processor;
    _voiceSource = source;
  }).catch(function() { stopVoice(); });
}

export function stopMicCapture() {
  setIsRecording(false);
  document.getElementById('mic-btn').classList.remove('recording');
  if (_voiceStream) { _voiceStream.getTracks().forEach(function(t) { t.stop(); }); _voiceStream = null; }
  if (_voiceProcessor) { _voiceProcessor.disconnect(); _voiceProcessor = null; }
  if (_voiceSource) { _voiceSource.disconnect(); _voiceSource = null; }
  var ctx = getAudioContext();
  if (ctx) { ctx.close(); setAudioContext(null); }
}

// ─── Wave visualisation ────────────────────────────────────────────

export function initWaveBars() {
  var wave = document.getElementById('voice-wave');
  wave.textContent = '';
  for (var i = 0; i < 24; i++) {
    var bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.height = '4px';
    wave.appendChild(bar);
  }
}

export function updateWave(rms) {
  var bars = document.querySelectorAll('#voice-wave .bar');
  bars.forEach(function(bar) {
    var h = Math.max(4, Math.min(40, rms * 800 * (0.5 + Math.random() * 0.5)));
    bar.style.height = h + 'px';
  });
}

// ─── TTS audio ─────────────────────────────────────────────────────

export function unlockTTSAudio() {
  try {
    var ctx = getTtsAudioCtx();
    if (!ctx || ctx.state === 'closed') {
      ctx = new AudioContext();
      setTtsAudioCtx(ctx);
    }
    if (ctx.state === 'suspended') {
      var p = ctx.resume();
      if (p && typeof p.catch === 'function') p.catch(function(){});
    }
  } catch (e) {}
}

export function getTTSAudioCtx(sr) {
  var ctx = getTtsAudioCtx();
  if (!ctx || ctx.state === 'closed') {
    ctx = new AudioContext({ sampleRate: sr });
    setTtsAudioCtx(ctx);
  }
  if (ctx.state === 'suspended') {
    var p = ctx.resume();
    if (p && typeof p.catch === 'function') p.catch(function(){});
  }
  return ctx;
}

/**
 * Direct queue for binary frames (no base64 decode needed).
 */
export function queueTTSChunkDirect(f32, sr) {
  var peak = 0;
  for (var i = 0; i < f32.length; i++) {
    var av = Math.abs(f32[i]);
    if (av > peak) peak = av;
  }
  if (peak > 0 && peak < 0.08) {
    var gain = Math.min(8.0, 0.18 / peak);
    for (var j = 0; j < f32.length; j++) f32[j] = Math.max(-1, Math.min(1, f32[j] * gain));
  }
  getTtsQueue().push({ f32: f32, sr: sr });
  if (getVoiceMode()) setVoiceState('speaking');
  if (!getTtsPlaying()) playNextChunk();
}

export function queueTTSChunk(b64, sr) {
  try {
    var bytes = Uint8Array.from(atob(b64), function(c) { return c.charCodeAt(0); });
    if (!bytes || bytes.byteLength < 4 || (bytes.byteLength % 4) !== 0) {
      throw new Error('invalid audio payload');
    }
    var n = bytes.byteLength / 4;
    var f32 = new Float32Array(n);
    var view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    var peak = 0;
    for (var i = 0; i < n; i++) {
      var v = view.getFloat32(i * 4, true);
      if (!Number.isFinite(v)) v = 0;
      f32[i] = v;
      var av = Math.abs(v);
      if (av > peak) peak = av;
    }
    if (peak > 0 && peak < 0.08) {
      var gain = Math.min(8.0, 0.18 / peak);
      for (var j = 0; j < f32.length; j++) f32[j] = Math.max(-1, Math.min(1, f32[j] * gain));
    }
    getTtsQueue().push({ f32: f32, sr: sr });
  } catch (e) {
    document.getElementById('status-tts').textContent = 'tts decode error';
    return;
  }
  if (getVoiceMode()) setVoiceState('speaking');
  if (!getTtsPlaying()) playNextChunk();
}

export function playNextChunk() {
  var queue = getTtsQueue();
  if (queue.length === 0) {
    setTtsPlaying(false);
    setTtsCurrentSource(null);
    if (getTtsDone()) onTTSFinished();
    return;
  }
  setTtsPlaying(true);
  var chunk = queue.shift();
  var ctx = getTTSAudioCtx(chunk.sr);
  function startChunk() {
    try {
      var buffer = ctx.createBuffer(1, chunk.f32.length, chunk.sr);
      buffer.getChannelData(0).set(chunk.f32);
      var src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      src.onended = function() { playNextChunk(); };
      setTtsCurrentSource(src);
      src.start(0);
    } catch (e) {
      setTtsCurrentSource(null);
      playNextChunk();
    }
  }
  if (ctx.state === 'suspended') {
    var p = ctx.resume();
    if (p && typeof p.then === 'function') {
      p.then(startChunk).catch(function() { playNextChunk(); });
      return;
    }
  }
  startChunk();
}

export function onTTSFinished() {
  setTtsDone(false);
  setTtsPlaying(false);
  getTtsQueue().length = 0;   // C4: in-place clearing
  setTtsCurrentSource(null);
  if (getVoiceMode()) {
    setVoiceState('listening');
    resumeVoiceListening();
  }
}
