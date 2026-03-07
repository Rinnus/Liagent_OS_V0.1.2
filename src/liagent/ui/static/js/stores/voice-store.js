/**
 * Voice/TTS state store.
 * Owns: voice overlay state, recording state, TTS queue/playback state.
 * C1: No DOM access — pure state + logic.
 * C4: In-place clearing for arrays (ttsQueue.length = 0).
 */

// ─── State ─────────────────────────────────────────────────────────
var voiceMode = false;
var voiceState = 'idle';          // idle | listening | hearing | processing | responding | speaking
var isRecording = false;
var audioContext = null;           // mic AudioContext
var ttsQueue = [];
var ttsPlaying = false;
var ttsAudioCtx = null;           // TTS AudioContext
var ttsDone = false;
var ttsCurrentSource = null;      // current AudioBufferSourceNode
var lastVoiceImageUrl = null;

// ─── Accessors ─────────────────────────────────────────────────────
export function getVoiceMode() { return voiceMode; }
export function setVoiceMode(v) { voiceMode = v; }

export function getVoiceState() { return voiceState; }
export function setVoiceState(v) { voiceState = v; }

export function getIsRecording() { return isRecording; }
export function setIsRecording(v) { isRecording = v; }

export function getAudioContext() { return audioContext; }
export function setAudioContext(v) { audioContext = v; }

export function getTtsQueue() { return ttsQueue; }

export function getTtsPlaying() { return ttsPlaying; }
export function setTtsPlaying(v) { ttsPlaying = v; }

export function getTtsAudioCtx() { return ttsAudioCtx; }
export function setTtsAudioCtx(v) { ttsAudioCtx = v; }

export function getTtsDone() { return ttsDone; }
export function setTtsDone(v) { ttsDone = v; }

export function getTtsCurrentSource() { return ttsCurrentSource; }
export function setTtsCurrentSource(v) { ttsCurrentSource = v; }

export function getLastVoiceImageUrl() { return lastVoiceImageUrl; }
export function setLastVoiceImageUrl(v) { lastVoiceImageUrl = v; }

// ─── Actions ───────────────────────────────────────────────────────

/**
 * Reset all TTS playback state (C4: in-place clearing).
 */
export function resetTTSState() {
  ttsQueue.length = 0;   // C4: in-place clearing
  ttsPlaying = false;
  ttsDone = false;
  if (ttsCurrentSource) {
    try { ttsCurrentSource.stop(); } catch(e) {}
    ttsCurrentSource = null;
  }
}

// ─── Audio helpers (pure logic, no DOM) ────────────────────────────

/**
 * Merge an array of Float32Array chunks into a single Float32Array.
 */
export function mergeFloat32Arrays(arrays) {
  var len = arrays.reduce(function(s, a) { return s + a.length; }, 0);
  var result = new Float32Array(len);
  var offset = 0;
  for (var i = 0; i < arrays.length; i++) { result.set(arrays[i], offset); offset += arrays[i].length; }
  return result;
}

/**
 * Convert a Float32Array to base64 string.
 */
export function float32ToBase64(f32) {
  var bytes = new Uint8Array(f32.buffer);
  var binary = '';
  for (var i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}
