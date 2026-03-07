/**
 * Media/camera/image state store.
 * Owns: pending image attachment, camera mode, camera stream/frame capture.
 * C1: No DOM access — pure state + logic.
 */

// ─── State ─────────────────────────────────────────────────────────
var pendingImage = null;         // { base64: string, dataUrl: string } or null
var cameraMode = false;
var cameraStream = null;
var cameraTimer = null;
var cameraFrameDataUrl = '';
var cameraFrameAtMs = 0;
var cameraCanvas = null;

// ─── Accessors ─────────────────────────────────────────────────────
export function getPendingImage() { return pendingImage; }
export function setPendingImage(v) { pendingImage = v; }

export function getCameraMode() { return cameraMode; }
export function setCameraMode(v) { cameraMode = v; }

export function getCameraStream() { return cameraStream; }
export function setCameraStream(v) { cameraStream = v; }

export function getCameraTimer() { return cameraTimer; }
export function setCameraTimer(v) { cameraTimer = v; }

export function getCameraFrameDataUrl() { return cameraFrameDataUrl; }
export function setCameraFrameDataUrl(v) { cameraFrameDataUrl = v; }

export function getCameraFrameAtMs() { return cameraFrameAtMs; }
export function setCameraFrameAtMs(v) { cameraFrameAtMs = v; }

export function getCameraCanvas() { return cameraCanvas; }
export function setCameraCanvas(v) { cameraCanvas = v; }
