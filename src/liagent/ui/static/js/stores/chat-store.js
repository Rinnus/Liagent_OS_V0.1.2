/**
 * Chat/message state store.
 * Owns: assistant element tracking, feedback, service tier/skill.
 * C1: No DOM access — pure state + logic.
 */

// ─── State ─────────────────────────────────────────────────────────
var currentAssistantEl = null;
var currentAssistantText = '';
var lastAssistantEl = null;
var lastAssistantRunId = '';
var feedbackTurnIndex = 0;
var lastUserQuery = '';
var lastToolUsed = '';
var activeServiceTier = '';
var activeSkillName = '';
var activeRunId = '';
var webSessionKey = '';

var WEB_SESSION_STORAGE_KEY = 'liagent_web_session_key';

function _buildWebSessionKey() {
  return 'web:' + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
}

function _loadWebSessionKey() {
  if (webSessionKey) return webSessionKey;
  try {
    webSessionKey = sessionStorage.getItem(WEB_SESSION_STORAGE_KEY) || '';
  } catch (e) {
    webSessionKey = '';
  }
  if (!webSessionKey) {
    webSessionKey = _buildWebSessionKey();
    try {
      sessionStorage.setItem(WEB_SESSION_STORAGE_KEY, webSessionKey);
    } catch (e) {}
  }
  return webSessionKey;
}

// ─── Accessors ─────────────────────────────────────────────────────
export function getCurrentAssistantEl() { return currentAssistantEl; }
export function setCurrentAssistantEl(el) { currentAssistantEl = el; }

export function getCurrentAssistantText() { return currentAssistantText; }
export function setCurrentAssistantText(t) { currentAssistantText = t; }

export function getLastAssistantEl() { return lastAssistantEl; }
export function setLastAssistantEl(el, runId) {
  lastAssistantEl = el;
  lastAssistantRunId = runId || '';
}
export function getLastAssistantRunId() { return lastAssistantRunId; }

export function getFeedbackTurnIndex() { return feedbackTurnIndex; }
export function setFeedbackTurnIndex(n) { feedbackTurnIndex = n; }

export function getLastUserQuery() { return lastUserQuery; }
export function setLastUserQuery(q) { lastUserQuery = q; }

export function getLastToolUsed() { return lastToolUsed; }
export function setLastToolUsed(t) { lastToolUsed = t; }

export function getActiveServiceTier() { return activeServiceTier; }
export function setActiveServiceTier(t) { activeServiceTier = t; }

export function getActiveSkillName() { return activeSkillName; }
export function setActiveSkillName(n) { activeSkillName = n; }

export function getActiveRunId() { return activeRunId; }
export function setActiveRunId(id) { activeRunId = id; }

export function getWebSessionKey() { return _loadWebSessionKey(); }
export function rotateWebSessionKey() {
  webSessionKey = _buildWebSessionKey();
  try {
    sessionStorage.setItem(WEB_SESSION_STORAGE_KEY, webSessionKey);
  } catch (e) {}
  return webSessionKey;
}

// ─── Clear ────────────────────────────────────────────────────────
export function clearChatState() {
  currentAssistantEl = null;
  currentAssistantText = '';
  lastAssistantEl = null;
  lastAssistantRunId = '';
  feedbackTurnIndex = 0;
  lastUserQuery = '';
  lastToolUsed = '';
  activeServiceTier = '';
  activeSkillName = '';
  activeRunId = '';
}

// ─── Shared utilities ──────────────────────────────────────────────
export function parseJSONSafe(text, fallback) {
  try { return JSON.parse(text); } catch (e) { return fallback; }
}

export function escHtml(s) {
  var d = document.createElement('div');
  d.appendChild(document.createTextNode(s));
  return d.innerHTML;
}

export function sanitizeHref(url) {
  var raw = String(url || '').trim();
  if (!raw) return '';
  try {
    var parsed = new URL(raw);
    var proto = parsed.protocol.toLowerCase();
    if (proto === 'https:') return raw;
    return '';
  } catch (e) { return ''; }
}
