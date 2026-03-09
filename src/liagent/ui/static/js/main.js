/**
 * Main entry point — imports all modules, binds event listeners, initializes.
 * Phase 2 complete: all code in ES modules, no inline script.
 */
import { connect } from './ws-client.js';
import {
  renderRunTimeline, clearRunTimelineAndRender,
  setTimelineLiveMode, clearArtifactSelection, clearAgentSelection,
  exportArtifacts,
} from './renderers/run-panel.js';
import {
  setTimelineSourceFilter, setTimelineEventFilter,
  setArtifactKindFilter, setArtifactSourceFilter, setArtifactTextFilter,
  getActiveTimelineRunId, getRunEventStore,
} from './stores/run-store.js';
import {
  toggleSettings, toggleTTS, clearChat,
  applyApiOnlyPreset, applyRuntimeConfig, applyLLM, applyTTS, applySTT,
  applyToolPolicy, applyLlmProviderPreset,
  toggleLLMFields, toggleTTSFields, toggleSTTFields,
  loadPolicyAudit, loadWeeklyMetrics,
  showAddMCPForm, sendMessage,
  cancelCurrentRun,
} from './renderers/settings-panel.js';
import { toggleVoice, stopVoice } from './renderers/voice-overlay.js';
import {
  toggleCameraMode, stopCameraMode,
  handleImageSelect, handleClipboardImage, clearPendingImage,
} from './utils/media-capture.js';

// ─── Run panel event bindings ──────────────────────────────────────
document.getElementById('run-panel-live').addEventListener('click', function() {
  setTimelineLiveMode(true);
});
document.getElementById('run-panel-clear').addEventListener('click', clearRunTimelineAndRender);
document.getElementById('run-artifact-close').addEventListener('click', clearArtifactSelection);
document.getElementById('run-agent-close').addEventListener('click', clearAgentSelection);
document.getElementById('run-agent-drawer-close').addEventListener('click', clearAgentSelection);
document.getElementById('timeline-source-filter').addEventListener('change', function() {
  setTimelineSourceFilter(String(this.value || '').trim());
  var rid = getActiveTimelineRunId();
  if (rid && getRunEventStore()[rid]) renderRunTimeline(rid);
});
document.getElementById('timeline-event-filter').addEventListener('change', function() {
  setTimelineEventFilter(String(this.value || '').trim());
  var rid = getActiveTimelineRunId();
  if (rid && getRunEventStore()[rid]) renderRunTimeline(rid);
});
document.getElementById('artifact-kind-filter').addEventListener('change', function() {
  setArtifactKindFilter(String(this.value || '').trim());
  var rid = getActiveTimelineRunId();
  if (rid && getRunEventStore()[rid]) renderRunTimeline(rid);
});
document.getElementById('artifact-source-filter').addEventListener('change', function() {
  setArtifactSourceFilter(String(this.value || '').trim());
  var rid = getActiveTimelineRunId();
  if (rid && getRunEventStore()[rid]) renderRunTimeline(rid);
});
document.getElementById('artifact-text-filter').addEventListener('input', function() {
  setArtifactTextFilter(String(this.value || '').trim().toLowerCase());
  var rid = getActiveTimelineRunId();
  if (rid && getRunEventStore()[rid]) renderRunTimeline(rid);
});
document.getElementById('artifact-export-btn').addEventListener('click', exportArtifacts);

// ─── Chat event bindings ───────────────────────────────────────────
document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('cancel-btn').addEventListener('click', cancelCurrentRun);
document.getElementById('btn-clear').addEventListener('click', clearChat);
document.getElementById('user-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
document.getElementById('user-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 160) + 'px';
});

// ─── Voice event bindings ──────────────────────────────────────────
document.getElementById('mic-btn').addEventListener('click', toggleVoice);
document.getElementById('voice-close-btn').addEventListener('click', stopVoice);
document.getElementById('btn-tts').addEventListener('click', toggleTTS);

// ─── Camera/Image event bindings ───────────────────────────────────
document.getElementById('camera-btn').addEventListener('click', toggleCameraMode);
document.getElementById('voice-cam-btn').addEventListener('click', toggleCameraMode);
document.getElementById('cam-stop-btn').addEventListener('click', stopCameraMode);
document.getElementById('img-btn').addEventListener('click', function() {
  document.getElementById('img-file-input').click();
});
document.getElementById('voice-img-btn').addEventListener('click', function() {
  document.getElementById('img-file-input').click();
});
document.getElementById('remove-img-btn').addEventListener('click', clearPendingImage);
document.getElementById('img-file-input').addEventListener('change', handleImageSelect);
document.addEventListener('paste', handleClipboardImage);
window.addEventListener('beforeunload', function() { stopCameraMode(); });

// ─── Settings event bindings ───────────────────────────────────────
document.getElementById('btn-settings').addEventListener('click', toggleSettings);

// Click outside settings panel to close it
document.addEventListener('click', function(e) {
  var panel = document.getElementById('settings-panel');
  if (!panel.classList.contains('open')) return;
  var btn = document.getElementById('btn-settings');
  if (panel.contains(e.target) || btn.contains(e.target)) return;
  panel.classList.remove('open');
});
document.getElementById('llm-backend').addEventListener('change', function() {
  toggleLLMFields();
});
document.getElementById('llm-provider').addEventListener('change', function() {
  applyLlmProviderPreset(false);
});
document.getElementById('tts-backend').addEventListener('change', function() {
  toggleTTSFields();
});
document.getElementById('stt-backend').addEventListener('change', function() {
  toggleSTTFields();
});
document.getElementById('apply-api-only').addEventListener('click', applyApiOnlyPreset);
document.getElementById('apply-runtime').addEventListener('click', applyRuntimeConfig);
document.getElementById('apply-llm').addEventListener('click', applyLLM);
document.getElementById('apply-tts').addEventListener('click', applyTTS);
document.getElementById('apply-stt').addEventListener('click', applySTT);
document.getElementById('apply-tool-policy').addEventListener('click', applyToolPolicy);
document.getElementById('refresh-audit').addEventListener('click', loadPolicyAudit);
document.getElementById('refresh-metrics').addEventListener('click', loadWeeklyMetrics);
document.getElementById('add-mcp-server').addEventListener('click', showAddMCPForm);
document.getElementById('tts-speed').addEventListener('input', function() {
  document.getElementById('tts-speed-val').textContent = Number(this.value || 1).toFixed(2);
});

// Lightweight speaker/speed switch — no model reload
document.getElementById('tts-speaker-name').addEventListener('change', function() {
  var self = this;
  import('./utils/auth.js').then(function(auth) {
    fetch('/api/config/tts_voice', {
      method: 'POST',
      headers: auth.withAuth({'Content-Type':'application/json'}),
      body: JSON.stringify({ speaker_name: self.value })
    }).then(function(r) { return r.json(); }).then(function(d) {
      if (d.tts_status) document.getElementById('status-tts').textContent = d.tts_status;
    });
  });
});
document.getElementById('tts-speed').addEventListener('change', function() {
  var self = this;
  import('./utils/auth.js').then(function(auth) {
    fetch('/api/config/tts_voice', {
      method: 'POST',
      headers: auth.withAuth({'Content-Type':'application/json'}),
      body: JSON.stringify({ speed: parseFloat(self.value || '1.0') })
    }).then(function(r) { return r.json(); }).then(function(d) {
      if (d.tts_status) document.getElementById('status-tts').textContent = d.tts_status;
    });
  });
});

// Thinking toggles
document.getElementById('toggle-show-thinking').addEventListener('click', function() {
  this.classList.toggle('on');
  import('./utils/auth.js').then(function(auth) {
    fetch('/api/config/thinking', {
      method: 'POST',
      headers: auth.withAuth({'Content-Type':'application/json'}),
      body: JSON.stringify({ show_thinking: document.getElementById('toggle-show-thinking').classList.contains('on') })
    });
  });
});
document.getElementById('toggle-enable-thinking').addEventListener('click', function() {
  this.classList.toggle('on');
  import('./utils/auth.js').then(function(auth) {
    fetch('/api/config/thinking', {
      method: 'POST',
      headers: auth.withAuth({'Content-Type':'application/json'}),
      body: JSON.stringify({ enable_thinking: document.getElementById('toggle-enable-thinking').classList.contains('on') })
    });
  });
});
document.getElementById('toggle-mcp-discovery').addEventListener('click', function() {
  this.classList.toggle('on');
});

// ─── Init ──────────────────────────────────────────────────────────
renderRunTimeline('');
connect();
