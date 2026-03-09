/**
 * WebSocket client — Phase 2 complete.
 * C2: connect() called exactly once (single-init guard).
 * C5: REGISTERED_MESSAGE_TYPES list for coverage validation.
 * Owns full handleWSMessage dispatch — no window._ bridges.
 */
import { trackRunEvent, clearRunTimeline } from './stores/run-store.js';
import { scheduleRenderRunTimeline, renderRunHistory, clearRunTimelineAndRender } from './renderers/run-panel.js';
import { setRunStatus } from './renderers/status-bar.js';
import {
  clearChatState,
  getCurrentAssistantEl, setCurrentAssistantEl,
  getCurrentAssistantText, setCurrentAssistantText,
  getFeedbackTurnIndex, setFeedbackTurnIndex,
  getLastAssistantEl, getLastAssistantRunId, setLastAssistantEl,
  getLastUserQuery, setLastUserQuery,
  getLastToolUsed, setLastToolUsed,
  getActiveServiceTier, setActiveServiceTier,
  getActiveSkillName, setActiveSkillName,
  getActiveRunId, setActiveRunId,
  getWebSessionKey,
  parseJSONSafe, escHtml,
} from './stores/chat-store.js';
import {
  getVoiceMode, getVoiceState,
  getTtsPlaying, getTtsQueue, getTtsDone, setTtsDone,
  getLastVoiceImageUrl, setLastVoiceImageUrl,
  resetTTSState,
} from './stores/voice-store.js';
import {
  startAssistantMessage, scrollBottom, getLastTextEl,
  addMessage, sendFeedback,
} from './renderers/message-panel.js';
import {
  setVoiceState, resumeVoiceListening,
  queueTTSChunk, queueTTSChunkDirect, onTTSFinished,
} from './renderers/voice-overlay.js';
import { loadConfig, loadWeeklyMetrics, sendToolConfirm, sendToolConfirmWithForce } from './renderers/settings-panel.js';
import { getWs, setWs } from './ws-send.js';

var _connected = false;
var _confirmationExpiryTimers = new Map();
var _cancelFinalizeTimer = 0;
var _CANCEL_NOTE_MIN_VISIBLE_MS = 180;

function setCancelButtonState(state) {
  var btn = document.getElementById('cancel-btn');
  if (!btn) return;
  var normalized = state || 'idle';
  btn.dataset.state = normalized;
  btn.classList.toggle('active', normalized === 'active');
  btn.classList.toggle('pending', normalized === 'pending');
  btn.disabled = normalized === 'idle' || normalized === 'pending';
  if (normalized === 'pending') {
    btn.title = 'Cancelling current task...';
  } else if (normalized === 'active') {
    btn.title = 'Cancel current task';
  } else {
    btn.title = 'No running task';
  }
}

function _clearCancelFinalizeTimer() {
  if (_cancelFinalizeTimer) {
    clearTimeout(_cancelFinalizeTimer);
    _cancelFinalizeTimer = 0;
  }
}

function updateCancelNote(text, assistantEl) {
  var targetEl = assistantEl || getCurrentAssistantEl();
  if (!targetEl) return null;
  var body = targetEl.querySelector('.msg-body');
  if (!body) return null;
  var note = body.querySelector('.cancel-note');
  if (!note) {
    note = document.createElement('div');
    note.className = 'tool-info clr-muted cancel-note';
    body.appendChild(note);
  }
  note.textContent = text || '';
  note.dataset.phase = (text === 'Current run cancelled.') ? 'done' : 'pending';
  if (!note.dataset.shownAt) note.dataset.shownAt = String(Date.now());
  return note;
}

function _finalizeCancelState(assistantEl, runId) {
  var targetEl = assistantEl || getCurrentAssistantEl() || getLastAssistantEl();
  if (!targetEl) return;
  updateCancelNote('Current run cancelled.', targetEl);
  setLastAssistantEl(targetEl, runId || getActiveRunId() || '');
  if (getCurrentAssistantEl() === targetEl) {
    setCurrentAssistantEl(null);
    setCurrentAssistantText('');
  }
  scrollBottom();
}

function _scheduleCancelFinalization(assistantEl, runId) {
  var targetEl = assistantEl || getCurrentAssistantEl();
  var note = targetEl ? targetEl.querySelector('.cancel-note') : null;
  var shownAt = note ? parseInt(note.dataset.shownAt || '0', 10) : 0;
  var elapsed = shownAt > 0 ? Math.max(0, Date.now() - shownAt) : _CANCEL_NOTE_MIN_VISIBLE_MS;
  var remaining = Math.max(0, _CANCEL_NOTE_MIN_VISIBLE_MS - elapsed);
  _clearCancelFinalizeTimer();
  if (!targetEl || !note || remaining <= 0) {
    _finalizeCancelState(targetEl, runId);
    return;
  }
  _cancelFinalizeTimer = setTimeout(function() {
    _cancelFinalizeTimer = 0;
    _finalizeCancelState(targetEl, runId);
  }, remaining);
}

function clearCancelNote() {
  _clearCancelFinalizeTimer();
  var assistantEl = getCurrentAssistantEl() || getLastAssistantEl();
  if (!assistantEl) return;
  var body = assistantEl.querySelector('.msg-body');
  if (!body) return;
  var note = body.querySelector('.cancel-note');
  if (note) note.remove();
}

function clearConfirmationExpiryTimer(token) {
  var key = String(token || '');
  if (!key) return;
  var timerId = _confirmationExpiryTimers.get(key);
  if (timerId) {
    clearTimeout(timerId);
    _confirmationExpiryTimers.delete(key);
  }
}

function scheduleConfirmationExpiry(token, expiresAt, blockEl, buttons) {
  var key = String(token || '');
  var expiryText = String(expiresAt || '');
  if (!key || !expiryText || !blockEl) return;
  clearConfirmationExpiryTimer(key);
  var expiresMs = Date.parse(expiryText);
  if (!Number.isFinite(expiresMs)) return;
  var fireExpiry = function() {
    _confirmationExpiryTimers.delete(key);
    if (!blockEl || !blockEl.isConnected) return;
    if (blockEl.dataset.confirmResolved === '1') return;
    blockEl.dataset.confirmResolved = '1';
    (buttons || []).forEach(function(btn) {
      if (btn) btn.disabled = true;
    });
    var note = blockEl.querySelector('.confirm-expired');
    if (!note) {
      note = document.createElement('div');
      note.className = 'tool-info clr-error confirm-expired';
      blockEl.appendChild(note);
    }
    note.textContent = 'Confirmation expired. Start the action again if you still want to run it.';
    scrollBottom();
  };
  var delayMs = Math.max(0, expiresMs - Date.now());
  var timerId = setTimeout(fireExpiry, delayMs);
  _confirmationExpiryTimers.set(key, timerId);
}

// Exported for coverage validation (C5)
export var REGISTERED_MESSAGE_TYPES = [
  'vision_input', 'vision_note', 'skill_selected', 'service_tier',
  'run_state', 'think', 'token', 'tool_start', 'tool_result', 'tool_error', 'tool_fallback', 'tool_skip',
  'llm_usage', 'context_update', 'dispatch', 'sub_complete',
  'synthesis_partial', 'synthesis',
  'policy_review', 'policy_blocked',
  'confirmation_required', 'tool_confirm_result',
  'bridge_tts',
  'status_message',
  'tts_chunk', 'tts_done', 'tts_profile', 'tts_metrics',
  'consistency_score', 'stt_start', 'stt_result',
  'task_outcome', 'run_metrics', 'proactive_suggestion', 'quality_gate',
  'done', 'error', 'cleared', 'finalized', 'feedback_ack', 'task_result', 'heartbeat_confirm', 'barge_in_ack',
  'auth_ok',
];

// ─── Full message dispatch (Phase 2 — no bridges) ──────────────────

function handleWSMessage(data) {
  // 1. Track in run store
  var result = trackRunEvent(data);
  if (result.shouldRender) {
    scheduleRenderRunTimeline(result.runId);
  }
  if (result.runId) {
    renderRunHistory();
  }

  // 2. Dispatch to domain handlers
  switch (data.type) {
    case 'vision_input': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var vi = document.createElement('div');
      vi.className = 'tool-info clr-accent';
      vi.textContent = 'vision frame attached (' + (data.count || 1) + ')';
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(vi);
      scrollBottom();
      break;
    }

    case 'vision_note': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var vn = document.createElement('div');
      vn.className = 'tool-info clr-muted';
      vn.textContent = 'vision note: ' + (data.text || '');
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(vn);
      scrollBottom();
      break;
    }

    case 'skill_selected': {
      var s = parseJSONSafe((data && data.result) || '{}', {});
      setActiveSkillName(s.name || '');
      if (getCurrentAssistantEl() && s.name) {
        var si = document.createElement('div');
        si.className = 'tool-info clr-info';
        si.textContent = 'skill: ' + s.name + ' | conf ' +
          (typeof s.confidence === 'number' ? s.confidence.toFixed(2) : '0.00') +
          (s.reason ? (' | ' + s.reason) : '');
        getCurrentAssistantEl().querySelector('.msg-body').appendChild(si);
      }
      setRunStatus('streaming', data.run_id || getActiveRunId() || '');
      break;
    }

    case 'service_tier': {
      var tierObj = parseJSONSafe((data && data.result) || '{}', {});
      setActiveServiceTier(tierObj.tier || '');
      if (tierObj.skill) setActiveSkillName(tierObj.skill);
      setRunStatus('streaming', data.run_id || getActiveRunId() || '');
      if (getVoiceMode() && (tierObj.tier || '')) {
        var extra = '';
        if (tierObj.adapted) extra = ' | adaptive';
        document.getElementById('voice-status').textContent = 'speaking (' + tierObj.tier + extra + ')...';
      }
      if (!getVoiceMode() && tierObj.adapted && getCurrentAssistantEl()) {
        var tip = document.createElement('div');
        tip.className = 'tool-info clr-info';
        tip.textContent = 'service tier adapted: ' + (tierObj.base_tier || '-') + ' -> ' +
          (tierObj.tier || '-') + ' (' + (tierObj.adapt_reason || 'overload') + ')';
        getCurrentAssistantEl().querySelector('.msg-body').appendChild(tip);
      }
      break;
    }

    case 'run_state': {
      var runId = data.run_id || '';
      if (runId) setActiveRunId(runId);
      setRunStatus(data.state || 'idle', getActiveRunId());
      if (data.state === 'accepted' || data.state === 'queued' || data.state === 'streaming') {
        if ((document.getElementById('cancel-btn') || {}).dataset.state !== 'pending') {
          setCancelButtonState('active');
        }
      }
      if ((data.state === 'cancelled' || data.state === 'error') && getVoiceMode()) {
        resetTTSState();
        setVoiceState('listening');
        resumeVoiceListening();
      }
      if (data.state === 'cancelled' && getCurrentAssistantEl()) {
        _scheduleCancelFinalization(getCurrentAssistantEl(), getActiveRunId() || '');
      }
      if (data.state === 'done' || data.state === 'error' || data.state === 'cancelled') {
        setCancelButtonState('idle');
        if (data.state !== 'cancelled') clearCancelNote();
        setActiveRunId('');
        setActiveServiceTier('');
        setActiveSkillName('');
      }
      break;
    }

    case 'think': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var thinkBlock = document.createElement('div');
      thinkBlock.className = 'think-block';
      var thinkBtn = document.createElement('button');
      thinkBtn.className = 'think-toggle';
      thinkBtn.textContent = '\u25B6 thinking';
      thinkBtn.onclick = function() {
        thinkBlock.classList.toggle('open');
        thinkBtn.textContent = thinkBlock.classList.contains('open') ? '\u25BC thinking' : '\u25B6 thinking';
      };
      var thinkContent = document.createElement('div');
      thinkContent.className = 'think-content';
      thinkContent.textContent = data.text || '';
      thinkBlock.appendChild(thinkBtn);
      thinkBlock.appendChild(thinkContent);
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(thinkBlock);
      scrollBottom();
      break;
    }

    case 'token':
      if (!getCurrentAssistantEl()) startAssistantMessage();
      setCurrentAssistantText(getCurrentAssistantText() + data.text);
      getLastTextEl().textContent = getCurrentAssistantText();
      scrollBottom();
      if (getVoiceMode()) {
        document.getElementById('voice-text').textContent = getCurrentAssistantText().slice(-200);
        if (getVoiceState() === 'processing') setVoiceState('responding');
      }
      break;

    case 'tool_start': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      setLastToolUsed(data.name || '');
      var info = document.createElement('div');
      info.className = 'tool-info';
      var args = data.args ? Object.entries(data.args).map(function(p) { return p[0]+'='+p[1]; }).join(', ') : '';
      info.textContent = data.name + '(' + args + ')';
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(info);
      setCurrentAssistantText('');
      var nt = document.createElement('div');
      nt.className = 'msg-text';
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(nt);
      scrollBottom();
      break;
    }

    case 'tool_result': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var ti = document.createElement('div');
      ti.className = 'tool-info clr-dim';
      var previewText = String(data.result || '');
      ti.textContent = previewText.substring(0, 200);
      var body = getCurrentAssistantEl().querySelector('.msg-body');
      body.appendChild(ti);
      if (data.truncated) {
        var tm = document.createElement('div');
        tm.className = 'tool-info clr-muted';
        var totalChars = Number(data.total_chars || 0);
        tm.textContent = totalChars > 0
          ? ('tool output preview | showing first 200 chars of ' + totalChars)
          : 'tool output preview | truncated';
        body.appendChild(tm);
        var fullText = String(data.full_result || '');
        if (fullText) {
          var details = document.createElement('details');
          details.className = 'tool-info tool-result-details';
          var summary = document.createElement('summary');
          summary.textContent = 'Expand full output';
          var pre = document.createElement('pre');
          pre.className = 'tool-result-full';
          pre.textContent = fullText;
          details.appendChild(summary);
          details.appendChild(pre);
          body.appendChild(details);
        }
      }
      scrollBottom();
      break;
    }

    case 'tool_error': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var terr = document.createElement('div');
      terr.className = 'tool-info clr-error';
      var errorType = String(data.error_type || '').trim();
      terr.textContent = 'tool error: ' + (data.name || '-') +
        (errorType ? (' [' + errorType + ']') : '') +
        (data.error ? (' - ' + String(data.error).slice(0, 200)) : '');
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(terr);
      scrollBottom();
      break;
    }

    case 'tool_fallback': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var tf = document.createElement('div');
      tf.className = 'tool-info clr-info';
      tf.textContent = 'tool fallback: ' + (data.requested_name || '-') + ' -> ' + (data.effective_name || '-');
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(tf);
      scrollBottom();
      break;
    }

    case 'tool_skip': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var ts = document.createElement('div');
      ts.className = 'tool-info clr-muted';
      ts.textContent = 'tool skipped: ' + (data.name || '-') + ' - ' + (data.reason || 'skipped');
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(ts);
      scrollBottom();
      break;
    }

    case 'llm_usage': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var usage = (data && typeof data.usage === 'object' && data.usage) ? data.usage : {};
      var provider = usage.provider || '-';
      var pt = Number(usage.prompt_tokens || 0);
      var ct = Number(usage.completion_tokens || 0);
      var tt = Number(usage.total_tokens || 0);
      var cpt = Number(usage.cached_prompt_tokens || 0);
      var cost = Number(usage.estimated_cost_usd || 0);
      var est = usage.estimated ? ' | estimated' : '';
      var budgetTxt = '';
      if (typeof usage.turn_budget === 'number') {
        budgetTxt = ' | turn ' + Number(usage.turn_used || 0) + '/' + Number(usage.turn_budget || 0);
      }
      var cacheTxt = cpt > 0 ? (' | cached=' + cpt) : '';
      var costTxt = cost > 0 ? (' | cost=$' + cost.toFixed(5)) : '';
      var ui = document.createElement('div');
      ui.className = 'tool-info clr-tool';
      ui.textContent = 'llm usage: ' + provider + ' p/c/t=' + pt + '/' + ct + '/' + tt + cacheTxt + costTxt + est + budgetTxt;
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(ui);
      scrollBottom();
      break;
    }

    case 'context_update': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var ctxData = parseJSONSafe(data.data || '{}', {});
      var varTag = document.createElement('span');
      varTag.className = 'ctx-var-tag';
      varTag.textContent = '$' + (ctxData.var || '?') + ' \u2190 ' + (ctxData.tool || '?');
      varTag.title = ctxData.preview || '';
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(varTag);
      scrollBottom();
      break;
    }

    case 'dispatch': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var dispatchData = (data && data.data && typeof data.data === 'object') ? data.data : {};
      var di = document.createElement('div');
      di.className = 'tool-info clr-info';
      di.textContent = 'dispatch: ' + (dispatchData.mode || '-') +
        ' | workers=' + Number(dispatchData.worker_count || 0) +
        (dispatchData.query ? (' | query=' + String(dispatchData.query).slice(0, 80)) : '');
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(di);
      scrollBottom();
      break;
    }

    case 'sub_complete': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var subData = (data && data.data && typeof data.data === 'object') ? data.data : {};
      var subSummary = '';
      if (subData && typeof subData === 'object') {
        subSummary = String(subData.summary || subData.error || '');
      } else {
        subSummary = String(data.data || '');
      }
      var sci = document.createElement('div');
      sci.className = 'tool-info clr-muted';
      sci.textContent = 'sub-agent done: ' + subSummary.slice(0, 200);
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(sci);
      scrollBottom();
      break;
    }

    case 'synthesis_partial': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var ps = (data && data.data && typeof data.data === 'object') ? data.data : {};
      var pinfo = document.createElement('div');
      pinfo.className = 'tool-info clr-info';
      pinfo.textContent = 'partial synthesis ready | agents=' + Number(ps.agent_count || 0);
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(pinfo);
      scrollBottom();
      break;
    }

    case 'synthesis': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var syn = (data && data.data && typeof data.data === 'object') ? data.data : {};
      var si2 = document.createElement('div');
      si2.className = 'tool-info clr-info';
      var cLen = Array.isArray(syn.citations) ? syn.citations.length : 0;
      si2.textContent = 'lead synthesis ready' + (cLen > 0 ? (' | citations=' + cLen) : '');
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(si2);
      scrollBottom();
      break;
    }

    case 'policy_review': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var pr = document.createElement('div');
      pr.className = 'tool-info clr-muted';
      pr.textContent = 'policy review ' + data.tool + ': ' + (data.result || '');
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(pr);
      scrollBottom();
      break;
    }

    case 'policy_blocked': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var blocked = document.createElement('div');
      blocked.className = 'tool-info clr-error';
      blocked.textContent = 'policy blocked: ' + data.name + ' - ' + data.reason;
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(blocked);
      scrollBottom();
      break;
    }

    case 'confirmation_required': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var cr = document.createElement('div');
      cr.className = 'tool-info clr-warn';
      cr.dataset.confirmToken = String(data.token || '');
      cr.dataset.confirmResolved = '0';
      var brief = parseJSONSafe(data.brief || '', {});
      var stage = brief.stage || 1;
      var requiredStage = brief.required_stage || 1;
      cr.textContent = 'confirmation required for ' + data.tool + ': ' + data.reason +
        ' (stage ' + stage + '/' + requiredStage + ')';
      var actions = document.createElement('div');
      actions.className = 'confirm-actions';
      var yesBtn = document.createElement('button');
      yesBtn.className = 'confirm-btn';
      yesBtn.textContent = 'approve';
      var noBtn = document.createElement('button');
      noBtn.className = 'confirm-btn';
      noBtn.textContent = 'reject';
      yesBtn.onclick = function() {
        cr.dataset.confirmResolved = '1';
        clearConfirmationExpiryTimer(data.token);
        sendToolConfirmWithForce(data.token, true, false, [yesBtn, noBtn]);
      };
      noBtn.onclick = function() {
        cr.dataset.confirmResolved = '1';
        clearConfirmationExpiryTimer(data.token);
        sendToolConfirm(data.token, false, [yesBtn, noBtn]);
      };
      actions.appendChild(yesBtn);
      actions.appendChild(noBtn);
      cr.appendChild(actions);
      if (brief && brief.capability) {
        var d2 = document.createElement('div');
        d2.style.marginTop = '6px';
        d2.style.fontSize = '11px';
        d2.textContent = 'capability: ' + brief.capability;
        cr.appendChild(d2);
      }
      scheduleConfirmationExpiry(data.token, data.expires_at, cr, [yesBtn, noBtn]);
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(cr);
      scrollBottom();
      break;
    }

    case 'tool_confirm_result': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var tcr = document.createElement('div');
      tcr.className = 'tool-info clr-muted';
      var tcResult = data.result || {};
      tcr.textContent = 'confirm result: ' + (tcResult.status || 'ok') + (tcResult.message ? (' - ' + tcResult.message) : '');
      if (tcResult.status === 'need_second_confirm') {
        var actions2 = document.createElement('div');
        actions2.className = 'confirm-actions';
        var finalBtn = document.createElement('button');
        finalBtn.className = 'confirm-btn';
        finalBtn.textContent = 'approve final';
        var cancelBtn = document.createElement('button');
        cancelBtn.className = 'confirm-btn';
        cancelBtn.textContent = 'reject';
        tcr.dataset.confirmToken = String(tcResult.token || '');
        tcr.dataset.confirmResolved = '0';
        finalBtn.onclick = function() {
          tcr.dataset.confirmResolved = '1';
          clearConfirmationExpiryTimer(tcResult.token);
          sendToolConfirmWithForce(tcResult.token, true, true, [finalBtn, cancelBtn]);
        };
        cancelBtn.onclick = function() {
          tcr.dataset.confirmResolved = '1';
          clearConfirmationExpiryTimer(tcResult.token);
          sendToolConfirmWithForce(tcResult.token, false, false, [finalBtn, cancelBtn]);
        };
        actions2.appendChild(finalBtn);
        actions2.appendChild(cancelBtn);
        tcr.appendChild(actions2);
        scheduleConfirmationExpiry(tcResult.token, tcResult.expires_at, tcr, [finalBtn, cancelBtn]);
      }
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(tcr);
      scrollBottom();
      break;
    }

    case 'tts_profile':
      if (getVoiceMode()) {
        setVoiceState('speaking');
        document.getElementById('voice-status').textContent = 'speaking (' + data.profile + ')...';
      }
      break;

    case 'tts_metrics':
      if (getVoiceMode()) {
        document.getElementById('voice-text').textContent =
          'first audio ' + (data.first_chunk_ms || 0) + 'ms | chunks ' + (data.chunks || 0);
      }
      break;

    case 'consistency_score':
      if (getVoiceMode()) {
        document.getElementById('voice-text').textContent = 'consistency score: ' + data.score;
      }
      break;

    case 'task_outcome': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var outcome = {};
      if (data && typeof data.result === 'string') {
        outcome = parseJSONSafe(data.result, {});
      } else if (data && typeof data.result === 'object' && data.result) {
        outcome = data.result;
      }
      var oi = document.createElement('div');
      oi.className = 'tool-info ' + (outcome.success ? 'clr-success' : 'clr-error');
      oi.textContent = 'task outcome: ' + (outcome.success ? 'success' : 'partial/failed') +
        (outcome.reason ? (' | ' + outcome.reason) : '');
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(oi);
      loadWeeklyMetrics();
      scrollBottom();
      break;
    }

    case 'run_metrics':
      if (getVoiceMode()) {
        document.getElementById('voice-text').textContent =
          'queue ' + (data.queued_ms || 0) + 'ms | stream ' + (data.stream_ms || 0) +
          'ms | tts ' + (data.tts_ms || 0) + 'ms';
      }
      loadWeeklyMetrics();
      break;

    case 'proactive_suggestion': {
      if (!getCurrentAssistantEl()) startAssistantMessage();
      var ps = document.createElement('div');
      ps.className = 'tool-info clr-accent';
      ps.textContent = data.message || '';
      var psActions = document.createElement('div');
      psActions.className = 'confirm-actions';
      var acceptBtn = document.createElement('button');
      acceptBtn.className = 'confirm-btn';
      acceptBtn.textContent = 'accept';
      var dismissBtn = document.createElement('button');
      dismissBtn.className = 'confirm-btn';
      dismissBtn.textContent = 'dismiss';
      var sugId = data.suggestion_id;
      var suggestionSessionKey = data.target_session_id || getWebSessionKey();
      acceptBtn.onclick = function() {
        var ws = getWs();
        if (ws && ws.readyState === 1) {
          ws.send(JSON.stringify({
            type: 'suggestion_feedback',
            id: sugId,
            action: 'accept',
            session_key: suggestionSessionKey
          }));
        }
        acceptBtn.disabled = true;
        dismissBtn.disabled = true;
        ps.style.opacity = '0.5';
      };
      dismissBtn.onclick = function() {
        var ws = getWs();
        if (ws && ws.readyState === 1) {
          ws.send(JSON.stringify({
            type: 'suggestion_feedback',
            id: sugId,
            action: 'dismiss',
            session_key: suggestionSessionKey
          }));
        }
        acceptBtn.disabled = true;
        dismissBtn.disabled = true;
        ps.style.opacity = '0.5';
      };
      psActions.appendChild(acceptBtn);
      psActions.appendChild(dismissBtn);
      ps.appendChild(psActions);
      getCurrentAssistantEl().querySelector('.msg-body').appendChild(ps);
      scrollBottom();
      break;
    }

    case 'quality_gate': {
      // Grounding gate verdict from orchestrator (post-answer quality check)
      var qualityTargetEl = getCurrentAssistantEl();
      if (!qualityTargetEl && data.run_id && getLastAssistantRunId() === data.run_id) {
        qualityTargetEl = getLastAssistantEl();
      }
      if (qualityTargetEl && data.verdict && data.verdict !== 'accept') {
        var qg = document.createElement('div');
        qg.className = 'tool-info clr-dim';
        qg.textContent = 'quality gate: ' + data.verdict;
        qualityTargetEl.querySelector('.msg-body').appendChild(qg);
        scrollBottom();
      }
      break;
    }

    case 'done':
      clearCancelNote();
      var completedAssistantEl = getCurrentAssistantEl();
      if (completedAssistantEl) {
        getLastTextEl().textContent = getCurrentAssistantText() || data.text;
        // Add feedback buttons
        var fbBar = document.createElement('div');
        fbBar.className = 'feedback-bar';
        var turnIdx = getFeedbackTurnIndex();
        setFeedbackTurnIndex(turnIdx + 1);
        var btnUp = document.createElement('button');
        btnUp.className = 'fb-btn';
        btnUp.dataset.turn = turnIdx;
        btnUp.dataset.fb = 'positive';
        btnUp.textContent = '\uD83D\uDC4D';
        btnUp.setAttribute('aria-label', 'Good response');
        btnUp.onclick = function() { sendFeedback(this); };
        var btnDown = document.createElement('button');
        btnDown.className = 'fb-btn';
        btnDown.dataset.turn = turnIdx;
        btnDown.dataset.fb = 'negative';
        btnDown.textContent = '\uD83D\uDC4E';
        btnDown.setAttribute('aria-label', 'Bad response');
        btnDown.onclick = function() { sendFeedback(this); };
        fbBar.appendChild(btnUp);
        fbBar.appendChild(btnDown);
        fbBar.dataset.query = getLastUserQuery();
        fbBar.dataset.answer = (getCurrentAssistantText() || data.text || '').slice(0, 500);
        fbBar.dataset.toolUsed = getLastToolUsed();
        fbBar.dataset.sessionKey = getWebSessionKey();
        if (data.quality && data.quality.confidence_label) {
          var badge = document.createElement('span');
          badge.className = 'confidence-badge confidence-' + data.quality.confidence_label;
          badge.textContent = data.quality.confidence_label;
          if (data.quality.confidence_note) {
            badge.title = data.quality.confidence_note;
          }
          fbBar.appendChild(badge);
        }
        completedAssistantEl.querySelector('.msg-body').appendChild(fbBar);
      }
      setLastAssistantEl(completedAssistantEl, data.run_id || getActiveRunId() || '');
      setCurrentAssistantEl(null);
      setCurrentAssistantText('');
      scrollBottom();
      if (!data.voice_pending && !getTtsPlaying() && getTtsQueue().length === 0 && getVoiceMode()) {
        setVoiceState('listening');
        resumeVoiceListening();
      }
      break;

    case 'bridge_tts':
      // Bridge phrase synthesized server-side; text shown as status
      if (data.text) {
        document.getElementById('status-tts').textContent = 'bridge: ' + data.text;
      }
      break;

    case 'status_message': {
      var statusEl = addMessage('assistant', data.text || '');
      statusEl.classList.add('assistant-status');
      var roleEl = statusEl.querySelector('.msg-role');
      if (roleEl) roleEl.textContent = 'status';
      var body = statusEl.querySelector('.msg-body');
      if (body && data.run_state) {
        var meta = document.createElement('div');
        meta.className = 'tool-info clr-muted';
        meta.textContent = 'run state: ' + String(data.run_state || 'streaming');
        body.appendChild(meta);
      }
      scrollBottom();
      break;
    }

    case 'tts_chunk':
      queueTTSChunk(data.audio, data.sample_rate || 24000);
      if (!getVoiceMode()) {
        var p = (typeof data.peak === 'number') ? data.peak : '?';
        document.getElementById('status-tts').textContent = 'playing | peak ' + p;
      }
      break;

    case 'tts_done':
      setTtsDone(true);
      if (!getTtsPlaying() && getTtsQueue().length === 0) {
        onTTSFinished();
      }
      break;

    case 'error': {
      clearCancelNote();
      var errText = data.text || 'error';
      var isTTSError = errText.indexOf('tts error:') === 0;
      if (getCurrentAssistantEl()) {
        var el = getLastTextEl();
        el.textContent = errText;
        el.classList.add('clr-dim');
      } else if (!getVoiceMode()) {
        var errEl = addMessage('assistant', errText);
        errEl.querySelector('.msg-text').classList.add('clr-error');
      }
      setCurrentAssistantEl(null);
      setCurrentAssistantText('');
      if (!getVoiceMode() && isTTSError) {
        document.getElementById('status-tts').textContent = errText;
      }
      if (getVoiceMode()) {
        document.getElementById('voice-text').textContent = errText;
        setVoiceState('listening');
        resumeVoiceListening();
      }
      break;
    }

    case 'stt_start':
      setVoiceState('processing');
      break;

    case 'stt_result':
      document.getElementById('voice-text').textContent = data.text;
      if (data.text) {
        resetTTSState();
        setVoiceState('processing');
        addMessage('user', data.text, getLastVoiceImageUrl());
        setLastVoiceImageUrl(null);
      } else {
        setLastVoiceImageUrl(null);
        if (getVoiceMode()) {
          setVoiceState('listening');
          resumeVoiceListening();
        }
      }
      break;

    case 'cleared':
      clearCancelNote();
      document.getElementById('messages').textContent = '';
      clearRunTimelineAndRender();
      clearChatState();
      break;

    case 'finalized':
      // Session facts saved to long-term memory (no UI action needed)
      break;

    case 'feedback_ack':
      break;

    case 'task_result': {
      var isSuccess = data.status === 'success';
      var isPreempted = data.status === 'preempted';
      var taskEl = document.createElement('div');
      taskEl.className = 'msg assistant task-result';
      var av = document.createElement('div');
      av.className = 'msg-avatar';
      av.textContent = 'T';
      av.style.background = isSuccess ? '#2a7a4b' : (isPreempted ? '#9a6a12' : '#7a2a2a');
      var bd = document.createElement('div');
      bd.className = 'msg-body';
      var rl = document.createElement('div');
      rl.className = 'msg-role';
      rl.textContent = (data.task_name || 'task') + (data.trigger_event ? ' [' + data.trigger_event + ']' : '');
      bd.appendChild(rl);
      var tx = document.createElement('div');
      tx.className = 'msg-text';
      if (isSuccess) {
        tx.textContent = data.result || '(empty)';
      } else if (isPreempted) {
        tx.textContent = data.result || 'Preempted by foreground request.';
      } else {
        tx.textContent = 'Error: ' + (data.error || 'unknown');
      }
      bd.appendChild(tx);
      taskEl.appendChild(av);
      taskEl.appendChild(bd);
      document.getElementById('messages').appendChild(taskEl);
      scrollBottom();
      break;
    }

    case 'heartbeat_confirm': {
      var hcEl = document.createElement('div');
      hcEl.className = 'msg assistant task-result';
      var hcAv = document.createElement('div');
      hcAv.className = 'msg-avatar';
      hcAv.textContent = 'H';
      hcAv.style.background = '#b8860b';
      var hcBd = document.createElement('div');
      hcBd.className = 'msg-body';
      var hcRole = document.createElement('div');
      hcRole.className = 'msg-role';
      hcRole.textContent = 'Heartbeat Confirmation Required';
      hcBd.appendChild(hcRole);
      var hcDesc = document.createElement('div');
      hcDesc.className = 'msg-text';
      hcDesc.textContent = data.description + ' [' + data.risk_level + ' risk]';
      hcBd.appendChild(hcDesc);
      var hcTool = document.createElement('div');
      hcTool.style.fontSize = '11px';
      hcTool.style.marginTop = '4px';
      hcTool.textContent = 'Tool: ' + data.action_type + ' | Expires: ' + data.expires_at;
      hcBd.appendChild(hcTool);
      if (data.action_args) {
        var hcArgs = document.createElement('div');
        hcArgs.style.fontSize = '11px';
        hcArgs.textContent = 'Args: ' + JSON.stringify(data.action_args);
        hcBd.appendChild(hcArgs);
      }
      var hcActions = document.createElement('div');
      hcActions.className = 'confirm-actions';
      var hcApprove = document.createElement('button');
      hcApprove.className = 'confirm-btn';
      hcApprove.textContent = 'approve';
      var hcReject = document.createElement('button');
      hcReject.className = 'confirm-btn';
      hcReject.textContent = 'reject';
      hcApprove.onclick = function() {
        hcApprove.disabled = true;
        hcReject.disabled = true;
        fetch('/api/tasks/confirm', {
          method: 'POST',
          headers: (function(h) { var t = (window.__LIAGENT_TOKEN__ || localStorage.getItem('liagent_token') || '').trim(); if (t) h['x-liagent-token'] = t; return h; })({'Content-Type': 'application/json'}),
          body: JSON.stringify({token: data.token, approved: true})
        }).then(function(r) { return r.json(); }).then(function(j) {
          hcApprove.textContent = j.error ? 'expired' : 'approved';
        });
      };
      hcReject.onclick = function() {
        hcApprove.disabled = true;
        hcReject.disabled = true;
        fetch('/api/tasks/confirm', {
          method: 'POST',
          headers: (function(h) { var t = (window.__LIAGENT_TOKEN__ || localStorage.getItem('liagent_token') || '').trim(); if (t) h['x-liagent-token'] = t; return h; })({'Content-Type': 'application/json'}),
          body: JSON.stringify({token: data.token, approved: false})
        }).then(function(r) { return r.json(); }).then(function(j) {
          hcReject.textContent = j.error ? 'expired' : 'rejected';
        });
      };
      hcActions.appendChild(hcApprove);
      hcActions.appendChild(hcReject);
      hcBd.appendChild(hcActions);
      hcEl.appendChild(hcAv);
      hcEl.appendChild(hcBd);
      document.getElementById('messages').appendChild(hcEl);
      scrollBottom();
      break;
    }

    case 'barge_in_ack':
      setCancelButtonState('idle');
      break;

    case 'auth_ok':
      break;
  }
}

// ─── Connect (C2: single-init) ────────────────────────────────────
export function connect() {
  if (_connected) {
    console.warn('ws-client: connect() called more than once — ignoring');
    return;
  }
  _connected = true;
  _doConnect();
}

function _doConnect() {
  var proto = location.protocol === 'https:' ? 'wss' : 'ws';
  var token = (window.__LIAGENT_TOKEN__ || localStorage.getItem('liagent_token') || '').trim();
  var url = proto + '://' + location.host + '/ws/chat';
  var conn = new WebSocket(url);
  setWs(conn);
  var _authed = false;

  conn.onopen = function() {
    setRunStatus('idle', '');
    if (token) {
      conn.send(JSON.stringify({ type: 'auth', token: token }));
    } else {
      _authed = true;
      loadConfig();
    }
  };
  conn.onmessage = function(e) {
    if (typeof e.data === 'string') {
      var parsed;
      try { parsed = JSON.parse(e.data); }
      catch (err) { console.warn('ws-client: malformed JSON frame:', err.message); return; }
      if (!_authed && parsed.type === 'auth_ok') {
        _authed = true;
        loadConfig();
        return;
      }
      if (!_authed && parsed.type === 'error') {
        console.error('ws-client: auth failed:', parsed.text);
        return;
      }
      handleWSMessage(parsed);
    } else if (e.data instanceof Blob) {
      e.data.arrayBuffer().then(function(buf) {
        var f32 = new Float32Array(buf);
        if (f32.length > 0) queueTTSChunkDirect(f32, 24000);
      });
    }
  };
  conn.onclose = function() {
    _connected = false;
    setRunStatus('disconnected', '');
    setTimeout(_doConnect, 2000);
  };
  conn.onerror = function() {
    _connected = false;
  };
}
