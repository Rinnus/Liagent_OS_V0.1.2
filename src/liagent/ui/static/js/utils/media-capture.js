/**
 * Media/camera/image capture utilities.
 * Handles image selection, clipboard paste, camera start/stop, frame capture.
 * Reads stores via getters, writes to DOM + stores via setters.
 */
import {
  getPendingImage, setPendingImage,
  getCameraMode, setCameraMode,
  getCameraStream, setCameraStream,
  getCameraTimer, setCameraTimer,
  getCameraFrameDataUrl, setCameraFrameDataUrl,
  setCameraFrameAtMs,
  getCameraCanvas, setCameraCanvas,
} from '../stores/media-store.js';
import { getVoiceMode } from '../stores/voice-store.js';
import { addMessage } from '../renderers/message-panel.js';

// ─── Image selection / clipboard ────────────────────────────────────

export function handleImageSelect(e) {
  var file = e.target.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(ev) {
    setPendingImage({ base64: ev.target.result, dataUrl: ev.target.result });
    showImagePreview();
  };
  reader.readAsDataURL(file);
  e.target.value = '';
}

export function handleClipboardImage(e) {
  var items = (e.clipboardData && e.clipboardData.items) || [];
  for (var i = 0; i < items.length; i++) {
    if (items[i].type && items[i].type.indexOf('image/') === 0) {
      var file = items[i].getAsFile();
      if (!file) continue;
      var reader = new FileReader();
      reader.onload = function(ev) {
        setPendingImage({ base64: ev.target.result, dataUrl: ev.target.result });
        showImagePreview();
      };
      reader.readAsDataURL(file);
      break;
    }
  }
}

// ─── Preview DOM helpers ────────────────────────────────────────────

export function showImagePreview() {
  var img = getPendingImage();
  if (!img) return;
  // Main input preview
  document.getElementById('img-preview-thumb').src = img.dataUrl;
  document.getElementById('img-preview').classList.add('active');
  document.getElementById('img-btn').classList.add('has-image');
  // Voice overlay preview
  if (getVoiceMode()) {
    document.getElementById('voice-img-thumb').src = img.dataUrl;
    document.getElementById('voice-img-preview').classList.add('active');
    document.getElementById('voice-img-btn').classList.add('has-image');
  }
}

export function clearPendingImage() {
  setPendingImage(null);
  document.getElementById('img-preview').classList.remove('active');
  document.getElementById('img-btn').classList.remove('has-image');
  document.getElementById('voice-img-preview').classList.remove('active');
  document.getElementById('voice-img-btn').classList.remove('has-image');
}

// ─── Camera mode ────────────────────────────────────────────────────

export function toggleCameraMode() {
  if (getCameraMode()) { stopCameraMode(); return; }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    console.error('[LiAgent] getUserMedia not available — need HTTPS or localhost');
    addMessage('system', 'Camera requires HTTPS or localhost access.');
    return;
  }
  startCameraMode();
}

export function syncCameraUI() {
  var preview = document.getElementById('cam-preview');
  var voicePreview = document.getElementById('voice-cam-preview');
  var inputBtn = document.getElementById('camera-btn');
  var voiceBtn = document.getElementById('voice-cam-btn');
  if (getCameraMode()) {
    preview.classList.add('active');
    inputBtn.classList.add('active');
    voiceBtn.classList.add('active');
    if (getVoiceMode()) voicePreview.classList.add('active');
  } else {
    preview.classList.remove('active');
    voicePreview.classList.remove('active');
    inputBtn.classList.remove('active');
    voiceBtn.classList.remove('active');
  }
}

export function startCameraMode() {
  navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 5, max: 8 },
      facingMode: 'user',
    },
    audio: false,
  }).then(function(stream) {
    setCameraMode(true);
    setCameraStream(stream);
    var video = document.getElementById('cam-video');
    var voiceVideo = document.getElementById('voice-cam-video');
    video.srcObject = stream;
    voiceVideo.srcObject = stream;
    syncCameraUI();
    var timer = getCameraTimer();
    if (timer) clearInterval(timer);
    setCameraTimer(setInterval(captureCameraFrame, 1200));
    setTimeout(captureCameraFrame, 250);
  }).catch(function(err) {
    console.error('[LiAgent] Camera access failed:', err.name, err.message);
    // Retry with relaxed constraints if overconstrained
    if (err.name === 'OverconstrainedError' || err.name === 'ConstraintNotSatisfiedError') {
      navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then(function(stream) {
        setCameraMode(true);
        setCameraStream(stream);
        var video = document.getElementById('cam-video');
        var voiceVideo = document.getElementById('voice-cam-video');
        video.srcObject = stream;
        voiceVideo.srcObject = stream;
        syncCameraUI();
        var timer = getCameraTimer();
        if (timer) clearInterval(timer);
        setCameraTimer(setInterval(captureCameraFrame, 1200));
        setTimeout(captureCameraFrame, 250);
      }).catch(function(err2) {
        console.error('[LiAgent] Camera fallback also failed:', err2.name, err2.message);
        addMessage('system', 'Camera access failed: ' + err2.message);
        setCameraMode(false);
        setCameraStream(null);
        setCameraFrameDataUrl('');
        syncCameraUI();
      });
      return;
    }
    addMessage('system', 'Camera access failed: ' + err.message);
    setCameraMode(false);
    setCameraStream(null);
    setCameraFrameDataUrl('');
    syncCameraUI();
  });
}

export function stopCameraMode() {
  setCameraMode(false);
  setCameraFrameDataUrl('');
  setCameraFrameAtMs(0);
  var timer = getCameraTimer();
  if (timer) {
    clearInterval(timer);
    setCameraTimer(null);
  }
  var stream = getCameraStream();
  if (stream) {
    stream.getTracks().forEach(function(t) { t.stop(); });
    setCameraStream(null);
  }
  var video = document.getElementById('cam-video');
  var voiceVideo = document.getElementById('voice-cam-video');
  if (video) video.srcObject = null;
  if (voiceVideo) voiceVideo.srcObject = null;
  syncCameraUI();
}

export function captureCameraFrame() {
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
