/**
 * Message panel renderer — message creation, plan box, feedback.
 * C1: Reads stores via getters, writes to DOM only.
 * Never mutates another domain's store.
 */
import {
  getCurrentAssistantEl, setCurrentAssistantEl,
  getCurrentAssistantText, setCurrentAssistantText,
  getFeedbackTurnIndex,
  getLastUserQuery,
  getLastToolUsed,
  parseJSONSafe, escHtml,
} from '../stores/chat-store.js';

import { getWs } from '../ws-send.js';

// ─── DOM helpers ────────────────────────────────────────────────────

export function getLastTextEl() {
  var el = getCurrentAssistantEl();
  if (!el) return null;
  var texts = el.querySelectorAll('.msg-text');
  return texts[texts.length - 1] || null;
}

export function scrollBottom() {
  var m = document.getElementById('messages');
  m.scrollTop = m.scrollHeight;
}

// ─── Messages ───────────────────────────────────────────────────────

export function addMessage(role, text, imageDataUrl) {
  var el = document.createElement('div');
  el.className = 'msg ' + role;

  var avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = role === 'user' ? 'Y' : 'A';

  var body = document.createElement('div');
  body.className = 'msg-body';

  var roleLabel = document.createElement('div');
  roleLabel.className = 'msg-role';
  roleLabel.textContent = role === 'user' ? 'you' : 'agent';

  body.appendChild(roleLabel);

  // Show image thumbnail in message if provided
  if (imageDataUrl) {
    var img = document.createElement('img');
    img.className = 'msg-image';
    img.src = imageDataUrl;
    body.appendChild(img);
  }

  var msgText = document.createElement('div');
  msgText.className = 'msg-text';
  msgText.textContent = text;

  body.appendChild(msgText);
  el.appendChild(avatar);
  el.appendChild(body);
  document.getElementById('messages').appendChild(el);
  scrollBottom();
  return el;
}

export function startAssistantMessage() {
  setCurrentAssistantText('');
  setCurrentAssistantEl(addMessage('assistant', ''));
}

// ─── Feedback ───────────────────────────────────────────────────────

export function sendFeedback(btn) {
  var ws = getWs();
  if (!ws || ws.readyState !== WebSocket.OPEN || btn.disabled) return;
  var fb = btn.dataset.fb;
  var bar = btn.parentElement;
  var turnIdx = parseInt(btn.dataset.turn, 10) || 0;
  var query = bar.dataset.query || '';
  var answer = bar.dataset.answer || '';
  var toolUsed = bar.dataset.toolUsed || '';
  ws.send(JSON.stringify({
    type: 'feedback',
    feedback: fb,
    turn_index: turnIdx,
    query: query,
    answer: answer,
    tool_used: toolUsed || null,
    session_key: bar.dataset.sessionKey || ''
  }));
  // Disable both buttons, highlight the selected one
  var btns = bar.querySelectorAll('.fb-btn');
  for (var i = 0; i < btns.length; i++) {
    btns[i].disabled = true;
  }
  btn.classList.add(fb === 'positive' ? 'active-up' : 'active-down');
  bar.classList.add('visible');
}
