/**
 * Settings panel renderer — config loading, LLM/TTS/STT apply, MCP management.
 * C1: Reads stores via getters, writes to DOM only.
 * Never mutates another domain's store directly.
 */
import {
  getSettingsFeedbackTimer, setSettingsFeedbackTimer,
  getLlmProviderPresetOrder, getLlmProviderPresets, getMcpPresets,
  normalizeApiBaseUrl, setLlmProviderCatalog,
  inferLlmProvider, inferLlmFamily, formatSTTStatus,
  hasStoredApiKey, currentInputValue, keyForApply,
} from '../stores/settings-store.js';

import { withAuth, authHeaders } from '../utils/auth.js';
import { getWs } from '../ws-send.js';
import { clearRunTimeline } from '../stores/run-store.js';
import { clearRunTimelineAndRender } from './run-panel.js';
import { addMessage } from './message-panel.js';
import {
  clearChatState,
  getWebSessionKey,
  rotateWebSessionKey,
  setLastUserQuery,
  setLastToolUsed,
  escHtml,
} from '../stores/chat-store.js';
import {
  getPendingImage, getCameraMode,
  getCameraFrameDataUrl, getCameraFrameAtMs,
} from '../stores/media-store.js';
import { clearPendingImage, captureCameraFrame } from '../utils/media-capture.js';
import { unlockTTSAudio } from './voice-overlay.js';
import { resetTTSState } from '../stores/voice-store.js';

// ─── Local utility ──────────────────────────────────────────────────
function _esc(s) {
  var d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

// ─── Settings feedback ──────────────────────────────────────────────

export function showSettingsFeedback(text, isError) {
  var el = document.getElementById('settings-feedback');
  if (!el) return;
  el.className = 'settings-feedback ' + (isError ? 'err' : 'ok');
  el.textContent = text || '';
  if (getSettingsFeedbackTimer()) {
    clearTimeout(getSettingsFeedbackTimer());
  }
  setSettingsFeedbackTimer(setTimeout(function() {
    el.className = 'settings-feedback';
    el.textContent = '';
  }, 3200));
}

export function setApplyBusy(btnId, busy, busyText) {
  var btn = document.getElementById(btnId);
  if (!btn) return;
  if (busy) {
    if (!btn.dataset.label) btn.dataset.label = btn.textContent;
    btn.textContent = busyText || 'Saving...';
    btn.disabled = true;
    btn.style.opacity = '0.72';
    return;
  }
  if (btn.dataset.label) btn.textContent = btn.dataset.label;
  btn.disabled = false;
  btn.style.opacity = '';
}

// ─── Shared fetch utility ───────────────────────────────────────────

export function postConfigJson(url, body) {
  return fetch(url, {
    method: 'POST',
    headers: withAuth({'Content-Type':'application/json'}),
    body: JSON.stringify(body)
  }).then(function(r) {
    return r.json().catch(function() { return {}; }).then(function(data) {
      if (!r.ok || data.error) {
        throw new Error(data.error || ('HTTP ' + r.status));
      }
      return data;
    });
  });
}

// ─── LLM provider rendering ────────────────────────────────────────

export function renderLlmProviderOptions(preferredValue) {
  var sel = document.getElementById('llm-provider');
  if (!sel) return;
  var LLM_PROVIDER_PRESETS = getLlmProviderPresets();
  var LLM_PROVIDER_PRESET_ORDER = getLlmProviderPresetOrder();
  var current = String(preferredValue || sel.value || '').trim() || 'openai';
  sel.textContent = '';
  var keys = Object.keys(LLM_PROVIDER_PRESETS || {});
  var ordered = [];
  for (var i = 0; i < LLM_PROVIDER_PRESET_ORDER.length; i++) {
    var key = LLM_PROVIDER_PRESET_ORDER[i];
    if (keys.indexOf(key) >= 0) ordered.push(key);
  }
  for (var j = 0; j < keys.length; j++) {
    var k = keys[j];
    if (ordered.indexOf(k) < 0) ordered.push(k);
  }
  for (var idx = 0; idx < ordered.length; idx++) {
    var provider = ordered[idx];
    var opt = document.createElement('option');
    opt.value = provider;
    if (provider === 'custom') {
      opt.textContent = 'Custom';
    } else {
      var preset = LLM_PROVIDER_PRESETS[provider] || {};
      opt.textContent = String(preset.label || provider);
    }
    sel.appendChild(opt);
  }
  if (!LLM_PROVIDER_PRESETS[current]) current = 'custom';
  sel.value = current;
}

export function applyLlmProviderPreset(forceOverwrite) {
  var LLM_PROVIDER_PRESETS = getLlmProviderPresets();
  var provider = document.getElementById('llm-provider').value;
  var preset = LLM_PROVIDER_PRESETS[provider];
  if (!preset) return;

  var urlEl = document.getElementById('llm-api-url');
  var modelEl = document.getElementById('llm-api-model');
  var familyEl = document.getElementById('llm-model-family');
  var currentUrl = String(urlEl.value || '').trim();
  var currentModel = String(modelEl.value || '').trim();

  if (forceOverwrite || !currentUrl) urlEl.value = preset.api_base_url;
  if (forceOverwrite || !currentModel) modelEl.value = preset.api_model;
  if (familyEl) familyEl.value = preset.model_family;
}

// ─── Field toggles ──────────────────────────────────────────────────

export function toggleLLMFields() {
  var isApi = document.getElementById('llm-backend').value === 'api';
  document.getElementById('llm-path').style.display = isApi ? 'none' : '';
  document.getElementById('llm-provider').style.display = isApi ? '' : 'none';
  document.getElementById('llm-api-url').style.display = isApi ? '' : 'none';
  document.getElementById('llm-api-key').style.display = isApi ? '' : 'none';
  document.getElementById('llm-api-model').style.display = isApi ? '' : 'none';
  document.getElementById('llm-model-family').style.display = isApi ? '' : 'none';
  document.getElementById('llm-cache-mode').style.display = isApi ? '' : 'none';
  document.getElementById('llm-cache-ttl').style.display = isApi ? '' : 'none';
  if (isApi) {
    applyLlmProviderPreset(false);
    if (!String(document.getElementById('llm-model-family').value || '').trim()) {
      document.getElementById('llm-model-family').value = inferLlmFamily(
        document.getElementById('llm-api-model').value,
        document.getElementById('llm-api-url').value
      );
    }
    if (!String(document.getElementById('llm-cache-mode').value || '').trim()) {
      document.getElementById('llm-cache-mode').value = 'implicit';
    }
    if (!String(document.getElementById('llm-cache-ttl').value || '').trim()) {
      document.getElementById('llm-cache-ttl').value = '600';
    }
  }
}

export function toggleTTSFields() {
  var isApi = document.getElementById('tts-backend').value === 'api';
  document.getElementById('tts-path').style.display = isApi ? 'none' : '';
  document.getElementById('tts-language').style.display = isApi ? 'none' : '';
  document.getElementById('tts-speaker-name').style.display = isApi ? 'none' : '';
  document.getElementById('tts-speed').parentElement.style.display = isApi ? 'none' : '';
  document.getElementById('tts-chunk-strategy').style.display = isApi ? 'none' : '';
  document.getElementById('tts-max-chars').style.display = isApi ? 'none' : '';
  document.getElementById('tts-api-url').style.display = isApi ? '' : 'none';
  document.getElementById('tts-api-key').style.display = isApi ? '' : 'none';
  document.getElementById('tts-api-model').style.display = isApi ? '' : 'none';
  document.getElementById('tts-api-voice').style.display = isApi ? '' : 'none';
  if (isApi) {
    if (!String(document.getElementById('tts-api-url').value || '').trim()) {
      document.getElementById('tts-api-url').value = 'https://api.openai.com/v1';
    }
    if (!String(document.getElementById('tts-api-model').value || '').trim()) {
      document.getElementById('tts-api-model').value = 'tts-1';
    }
    if (!String(document.getElementById('tts-api-voice').value || '').trim()) {
      document.getElementById('tts-api-voice').value = 'alloy';
    }
  }
}

export function toggleSTTFields() {
  var isApi = document.getElementById('stt-backend').value === 'api';
  document.getElementById('stt-model').style.display = isApi ? 'none' : '';
  document.getElementById('stt-api-url').style.display = isApi ? '' : 'none';
  document.getElementById('stt-api-key').style.display = isApi ? '' : 'none';
  document.getElementById('stt-api-model').style.display = isApi ? '' : 'none';
  if (isApi) {
    if (!String(document.getElementById('stt-api-url').value || '').trim()) {
      document.getElementById('stt-api-url').value = 'https://api.openai.com/v1';
    }
    if (!String(document.getElementById('stt-api-model').value || '').trim()) {
      document.getElementById('stt-api-model').value = 'gpt-4o-mini-transcribe';
    }
  }
}

// ─── Apply functions ────────────────────────────────────────────────

export function applyLLM() {
  var LLM_PROVIDER_PRESETS = getLlmProviderPresets();
  setApplyBusy('apply-llm', true, 'Applying...');
  var backend = document.getElementById('llm-backend').value;
  var body = {};
  if (backend === 'local') {
    var path = String(document.getElementById('llm-path').value || '').trim();
    if (!path) {
      setApplyBusy('apply-llm', false);
      showSettingsFeedback('LLM local mode requires model path.', true);
      return;
    }
    body = { backend: 'local', local_model_path: path };
  } else {
    var llmModel = String(document.getElementById('llm-api-model').value || '').trim();
    var llmKey = String(document.getElementById('llm-api-key').value || '').trim();
    var llmUrl = normalizeApiBaseUrl(document.getElementById('llm-api-url').value);
    var inferredProvider = inferLlmProvider(llmModel, llmUrl);
    var inferredPreset = LLM_PROVIDER_PRESETS[inferredProvider] || {};
    if (!llmModel) {
      setApplyBusy('apply-llm', false);
      showSettingsFeedback('LLM API mode requires model name.', true);
      return;
    }
    if (!llmKey && !hasStoredApiKey('llm-api-key')) {
      setApplyBusy('apply-llm', false);
      showSettingsFeedback('LLM API mode requires API key (first setup).', true);
      return;
    }
    body = {
      backend: 'api',
      api_base_url: llmUrl,
      api_key: llmKey,
      api_model: llmModel,
      model_family: String(document.getElementById('llm-model-family').value || '').trim()
        || inferLlmFamily(llmModel, llmUrl),
      tool_protocol: String(inferredPreset.tool_protocol || 'openai_function'),
      api_cache_mode: String(document.getElementById('llm-cache-mode').value || 'implicit').trim() || 'implicit',
      api_cache_ttl_sec: Number(document.getElementById('llm-cache-ttl').value || 600) || 600
    };
  }
  postConfigJson('/api/config/llm', body).then(function(d) {
    if (d.llm_status) document.getElementById('status-llm').textContent = d.llm_status;
    showSettingsFeedback('LLM settings applied.', false);
    loadConfig();
  }).catch(function(e) {
    showSettingsFeedback('LLM apply failed: ' + (e && e.message ? e.message : 'unknown error'), true);
  }).finally(function() {
    setApplyBusy('apply-llm', false);
  });
}

export function applyTTS() {
  setApplyBusy('apply-tts', true, 'Applying...');
  var backend = document.getElementById('tts-backend').value;
  var body = {};
  if (backend === 'local') {
    var ttsPath = String(document.getElementById('tts-path').value || '').trim();
    if (!ttsPath) {
      setApplyBusy('apply-tts', false);
      showSettingsFeedback('TTS local mode requires model path.', true);
      return;
    }
    body = { backend: 'local', tts_engine: 'qwen3',
      local_model_path: ttsPath,
      language: document.getElementById('tts-language').value || 'zh',
      speaker_name: document.getElementById('tts-speaker-name').value || 'serena',
      speed: parseFloat(document.getElementById('tts-speed').value || '1.0'),
      chunk_strategy: document.getElementById('tts-chunk-strategy').value || 'smart_chunk',
      max_chunk_chars: parseInt(document.getElementById('tts-max-chars').value || '220', 10) };
  } else {
    var ttsModel = String(document.getElementById('tts-api-model').value || '').trim();
    var ttsKey = String(document.getElementById('tts-api-key').value || '').trim();
    if (!ttsModel) {
      setApplyBusy('apply-tts', false);
      showSettingsFeedback('TTS API mode requires model name.', true);
      return;
    }
    if (!ttsKey && !hasStoredApiKey('tts-api-key')) {
      setApplyBusy('apply-tts', false);
      showSettingsFeedback('TTS API mode requires API key (first setup).', true);
      return;
    }
    body = { backend: 'api',
      api_base_url: normalizeApiBaseUrl(document.getElementById('tts-api-url').value),
      api_key: ttsKey,
      api_model: ttsModel,
      api_voice: String(document.getElementById('tts-api-voice').value || '').trim() || 'alloy' };
  }
  postConfigJson('/api/config/tts', body).then(function(d) {
    if (d.tts_status) document.getElementById('status-tts').textContent = d.tts_status;
    showSettingsFeedback('TTS settings applied.', false);
    loadConfig();
  }).catch(function(e) {
    showSettingsFeedback('TTS apply failed: ' + (e && e.message ? e.message : 'unknown error'), true);
  }).finally(function() {
    setApplyBusy('apply-tts', false);
  });
}

export function applySTT() {
  setApplyBusy('apply-stt', true, 'Applying...');
  var backend = document.getElementById('stt-backend').value;
  var body = {};
  if (backend === 'local') {
    var sttPath = String(document.getElementById('stt-model').value || '').trim();
    if (!sttPath) {
      setApplyBusy('apply-stt', false);
      showSettingsFeedback('STT local mode requires model path.', true);
      return;
    }
    body = {
      backend: 'local',
      model: sttPath,
      language: document.getElementById('stt-language').value || 'auto'
    };
  } else {
    var sttModel = String(document.getElementById('stt-api-model').value || '').trim();
    var sttKey = String(document.getElementById('stt-api-key').value || '').trim();
    if (!sttModel) {
      setApplyBusy('apply-stt', false);
      showSettingsFeedback('STT API mode requires model name.', true);
      return;
    }
    if (!sttKey && !hasStoredApiKey('stt-api-key')) {
      setApplyBusy('apply-stt', false);
      showSettingsFeedback('STT API mode requires API key (first setup).', true);
      return;
    }
    body = {
      backend: 'api',
      language: document.getElementById('stt-language').value || 'auto',
      api_base_url: normalizeApiBaseUrl(document.getElementById('stt-api-url').value),
      api_key: sttKey,
      api_model: sttModel
    };
  }
  postConfigJson('/api/config/stt', body).then(function(d) {
    if (d.stt_status) document.getElementById('status-stt').textContent = d.stt_status;
    showSettingsFeedback('STT settings applied.', false);
    loadConfig();
  }).catch(function(e) {
    showSettingsFeedback('STT apply failed: ' + (e && e.message ? e.message : 'unknown error'), true);
  }).finally(function() {
    setApplyBusy('apply-stt', false);
  });
}

export function applyApiOnlyPreset() {
  setApplyBusy('apply-api-only', true, 'Applying...');

  document.getElementById('llm-backend').value = 'api';
  document.getElementById('tts-backend').value = 'api';
  document.getElementById('stt-backend').value = 'api';
  toggleLLMFields();
  toggleTTSFields();
  toggleSTTFields();

  var llmProvider = currentInputValue('llm-provider') || 'openai';
  if (llmProvider !== 'custom') {
    applyLlmProviderPreset(false);
  }
  var llmUrl = normalizeApiBaseUrl(document.getElementById('llm-api-url').value);
  var ttsUrl = normalizeApiBaseUrl(document.getElementById('tts-api-url').value);
  var sttUrl = normalizeApiBaseUrl(document.getElementById('stt-api-url').value);
  var llmModel = currentInputValue('llm-api-model') || 'gpt-4o';
  var llmFamily = currentInputValue('llm-model-family') || inferLlmFamily(llmModel, llmUrl);
  var llmCacheMode = currentInputValue('llm-cache-mode') || 'implicit';
  var llmCacheTtl = Number(currentInputValue('llm-cache-ttl') || 600) || 600;
  var ttsModel = currentInputValue('tts-api-model') || 'tts-1';
  var sttModel = currentInputValue('stt-api-model') || 'gpt-4o-mini-transcribe';
  var ttsVoice = currentInputValue('tts-api-voice') || 'alloy';
  var sharedKey = currentInputValue('llm-api-key') || currentInputValue('tts-api-key') || currentInputValue('stt-api-key');

  var llmHasKey = hasStoredApiKey('llm-api-key') || Boolean(currentInputValue('llm-api-key')) || Boolean(sharedKey);
  var ttsHasKey = hasStoredApiKey('tts-api-key') || Boolean(currentInputValue('tts-api-key')) || Boolean(sharedKey);
  var sttHasKey = hasStoredApiKey('stt-api-key') || Boolean(currentInputValue('stt-api-key')) || Boolean(sharedKey);
  if (!llmHasKey || !ttsHasKey || !sttHasKey) {
    setApplyBusy('apply-api-only', false);
    showSettingsFeedback('API-only setup needs at least one API key (or previously stored keys for each engine).', true);
    return;
  }

  var llmBody = {
    backend: 'api',
    api_base_url: llmUrl,
    api_key: keyForApply('llm-api-key', sharedKey),
    api_model: llmModel,
    model_family: llmFamily,
    api_cache_mode: llmCacheMode,
    api_cache_ttl_sec: llmCacheTtl
  };
  var ttsBody = {
    backend: 'api',
    api_base_url: ttsUrl,
    api_key: keyForApply('tts-api-key', sharedKey),
    api_model: ttsModel,
    api_voice: ttsVoice
  };
  var sttBody = {
    backend: 'api',
    language: document.getElementById('stt-language').value || 'auto',
    api_base_url: sttUrl,
    api_key: keyForApply('stt-api-key', sharedKey),
    api_model: sttModel
  };

  postConfigJson('/api/config/llm', llmBody)
    .then(function() { return postConfigJson('/api/config/tts', ttsBody); })
    .then(function() { return postConfigJson('/api/config/stt', sttBody); })
    .then(function() {
      return postConfigJson('/api/config/runtime', { runtime_mode: 'cloud_performance' });
    })
    .then(function() {
      showSettingsFeedback('API-only mode enabled (LLM/TTS/STT).', false);
      loadConfig();
      setTimeout(function() {
        document.getElementById('settings-panel').classList.remove('open');
      }, 800);
    })
    .catch(function(e) {
      showSettingsFeedback('API-only setup failed: ' + (e && e.message ? e.message : 'unknown error'), true);
    })
    .finally(function() {
      setApplyBusy('apply-api-only', false);
    });
}

export function applyRuntimeConfig() {
  setApplyBusy('apply-runtime', true, 'Applying...');
  var runtimeMode = String(document.getElementById('runtime-mode').value || 'hybrid_balanced').trim();
  var replMode = String(document.getElementById('repl-mode').value || 'sandboxed').trim();
  var sandboxMode = String(document.getElementById('sandbox-mode').value || 'off').trim();
  var sandboxImage = String(document.getElementById('sandbox-image').value || '').trim() || 'liagent-sandbox:latest';
  var softTokens = Number(document.getElementById('budget-soft-tokens').value || 60000) || 60000;
  var hardTokens = Number(document.getElementById('budget-hard-tokens').value || 120000) || 120000;
  var softUsd = Number(document.getElementById('budget-soft-usd').value || 1.5) || 1.5;
  var hardUsd = Number(document.getElementById('budget-hard-usd').value || 3.0) || 3.0;
  var mcpEnabled = document.getElementById('toggle-mcp-discovery').classList.contains('on');
  var mcpReloadSec = Number(document.getElementById('mcp-hot-reload-sec').value || 120) || 120;

  if (hardTokens < softTokens) hardTokens = softTokens;
  if (hardUsd < softUsd) hardUsd = softUsd;
  var confirmTrusted = false;
  if (replMode === 'trusted_local') {
    confirmTrusted = window.confirm('trusted_local disables REPL sandbox protections. Continue?');
    if (!confirmTrusted) {
      setApplyBusy('apply-runtime', false);
      return;
    }
  }

  postConfigJson('/api/config/runtime', {
    runtime_mode: runtimeMode,
    repl_mode: replMode,
    confirm_trusted_local: confirmTrusted,
    sandbox: {
      enabled: sandboxMode !== 'off',
      mode: sandboxMode,
      image: sandboxImage
    },
    budget: {
      session_soft_tokens: Math.max(1000, Math.floor(softTokens)),
      session_hard_tokens: Math.max(1000, Math.floor(hardTokens)),
      session_soft_usd: Math.max(0, softUsd),
      session_hard_usd: Math.max(0, hardUsd)
    },
    mcp_discovery: {
      enabled: mcpEnabled,
      hot_reload_sec: Math.max(30, Math.floor(mcpReloadSec))
    }
  }).then(function() {
    showSettingsFeedback('Runtime settings applied.', false);
    loadConfig();
  }).catch(function(e) {
    showSettingsFeedback('Runtime apply failed: ' + (e && e.message ? e.message : 'unknown error'), true);
  }).finally(function() {
    setApplyBusy('apply-runtime', false);
  });
}

export function applyToolPolicy() {
  var body = {
    tool_profile: document.getElementById('tool-profile').value || 'research',
  };
  fetch('/api/config/tool_policy', {
    method: 'POST',
    headers: withAuth({'Content-Type':'application/json'}),
    body: JSON.stringify(body),
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d && d.tool_profile) {
        document.getElementById('tool-profile').value = d.tool_profile;
      }
      showSettingsFeedback('Tool policy applied.', false);
      loadPolicyAudit();
      setTimeout(function() {
        document.getElementById('settings-panel').classList.remove('open');
      }, 800);
    });
}

// ─── Config loading ─────────────────────────────────────────────────

export function loadConfig() {
  fetch('/api/config', { headers: authHeaders() }).then(function(r) {
    return r.json().catch(function() { return {}; }).then(function(data) {
      if (!r.ok || data.error) {
        throw new Error(data.error || ('HTTP ' + r.status));
      }
      return data;
    });
  }).then(function(data) {
    document.getElementById('status-llm').textContent = data.llm_status;
    document.getElementById('status-tts').textContent = data.tts_status;
    document.getElementById('status-stt').textContent = data.stt_status || formatSTTStatus(data.config || {});
    var catalogChanged = setLlmProviderCatalog(data.llm_provider_catalog);
    renderLlmProviderOptions();
    document.getElementById('dot-tts').className = data.tts_enabled ? 'dot on' : 'dot';
    var cfg = data.config;
    var llmReady = (cfg.llm.backend === 'api')
      ? Boolean(cfg.llm.api_model && (cfg.llm.api_key_masked || cfg.llm.api_key))
      : Boolean(cfg.llm.local_model_path);
    document.getElementById('dot-llm').className = llmReady ? 'dot on' : 'dot';
    var sttBackend = ((cfg.stt && cfg.stt.backend) || 'local').toLowerCase();
    var sttReady = sttBackend === 'api'
      ? Boolean((cfg.stt && cfg.stt.api_model) && ((cfg.stt && cfg.stt.api_key_masked) || (cfg.stt && cfg.stt.api_key)))
      : Boolean(cfg.stt && cfg.stt.model);
    document.getElementById('dot-stt').className = sttReady ? 'dot on' : 'dot';
    document.getElementById('tool-profile').value = cfg.tool_profile || 'research';
    document.getElementById('runtime-mode').value = cfg.runtime_mode || 'hybrid_balanced';
    var replMode = String(cfg.repl_mode || 'sandboxed');
    if (!document.querySelector('#repl-mode option[value="' + replMode + '"]')) {
      replMode = 'sandboxed';
    }
    document.getElementById('repl-mode').value = replMode;
    var sandboxCfg = cfg.sandbox || {};
    var sandboxMode = String(sandboxCfg.mode || 'off');
    if (!document.querySelector('#sandbox-mode option[value="' + sandboxMode + '"]')) {
      sandboxMode = 'off';
    }
    document.getElementById('sandbox-mode').value = sandboxMode;
    document.getElementById('sandbox-image').value = String(sandboxCfg.image || 'liagent-sandbox:latest');
    var budgetCfg = cfg.budget || {};
    document.getElementById('budget-soft-tokens').value = String(budgetCfg.session_soft_tokens || 60000);
    document.getElementById('budget-hard-tokens').value = String(budgetCfg.session_hard_tokens || 120000);
    document.getElementById('budget-soft-usd').value = String(
      budgetCfg.session_soft_usd != null ? budgetCfg.session_soft_usd : 1.5
    );
    document.getElementById('budget-hard-usd').value = String(
      budgetCfg.session_hard_usd != null ? budgetCfg.session_hard_usd : 3.0
    );
    var mcpDiscovery = cfg.mcp_discovery || {};
    document.getElementById('toggle-mcp-discovery').className =
      mcpDiscovery.enabled ? 'mcp-toggle on' : 'mcp-toggle';
    document.getElementById('mcp-hot-reload-sec').value = String(mcpDiscovery.hot_reload_sec || 120);
    document.getElementById('llm-backend').value = cfg.llm.backend;
    document.getElementById('llm-path').value = cfg.llm.local_model_path;
    document.getElementById('llm-api-url').value = cfg.llm.api_base_url || 'https://api.openai.com/v1';
    document.getElementById('llm-api-key').value = '';
    document.getElementById('llm-api-key').placeholder = cfg.llm.api_key_masked ? ('Stored: ' + cfg.llm.api_key_masked) : 'API Key';
    document.getElementById('llm-api-model').value = cfg.llm.api_model || 'gpt-4o';
    var llmProvider = inferLlmProvider(cfg.llm.api_model, cfg.llm.api_base_url);
    if (!document.querySelector('#llm-provider option[value="' + llmProvider + '"]')) {
      llmProvider = 'custom';
    }
    document.getElementById('llm-provider').value = llmProvider;
    var llmFamily = String(cfg.llm.model_family || '').trim() || inferLlmFamily(cfg.llm.api_model, cfg.llm.api_base_url);
    if (!document.querySelector('#llm-model-family option[value="' + llmFamily + '"]')) {
      llmFamily = 'openai';
    }
    document.getElementById('llm-model-family').value = llmFamily;
    var llmCacheMode = String(cfg.llm.api_cache_mode || 'implicit');
    if (!document.querySelector('#llm-cache-mode option[value="' + llmCacheMode + '"]')) {
      llmCacheMode = 'implicit';
    }
    document.getElementById('llm-cache-mode').value = llmCacheMode;
    document.getElementById('llm-cache-ttl').value = String(cfg.llm.api_cache_ttl_sec || 600);
    document.getElementById('tts-backend').value = cfg.tts.backend;
    document.getElementById('tts-path').value = cfg.tts.local_model_path;
    document.getElementById('tts-language').value = cfg.tts.language || 'zh';
    document.getElementById('tts-speaker-name').value = cfg.tts.speaker_name || 'serena';
    var speed = Number(cfg.tts.speed || 1.0);
    document.getElementById('tts-speed').value = speed;
    document.getElementById('tts-speed-val').textContent = speed.toFixed(2);
    document.getElementById('tts-chunk-strategy').value = cfg.tts.chunk_strategy || 'smart_chunk';
    document.getElementById('tts-max-chars').value = cfg.tts.max_chunk_chars || 220;
    document.getElementById('tts-api-url').value = cfg.tts.api_base_url || 'https://api.openai.com/v1';
    document.getElementById('tts-api-key').value = '';
    document.getElementById('tts-api-key').placeholder = cfg.tts.api_key_masked ? ('Stored: ' + cfg.tts.api_key_masked) : 'API Key';
    document.getElementById('tts-api-model').value = cfg.tts.api_model || 'tts-1';
    document.getElementById('tts-api-voice').value = cfg.tts.api_voice || 'alloy';
    document.getElementById('stt-backend').value = (cfg.stt && cfg.stt.backend) || 'local';
    document.getElementById('stt-model').value = (cfg.stt && cfg.stt.model) || '';
    document.getElementById('stt-language').value = (cfg.stt && cfg.stt.language) || 'auto';
    document.getElementById('stt-api-url').value = (cfg.stt && cfg.stt.api_base_url) || 'https://api.openai.com/v1';
    document.getElementById('stt-api-key').value = '';
    document.getElementById('stt-api-key').placeholder =
      (cfg.stt && cfg.stt.api_key_masked) ? ('Stored: ' + cfg.stt.api_key_masked) : 'API Key';
    document.getElementById('stt-api-model').value = (cfg.stt && cfg.stt.api_model) || 'gpt-4o-mini-transcribe';
    // Thinking toggles
    document.getElementById('toggle-show-thinking').className = cfg.show_thinking ? 'mcp-toggle on' : 'mcp-toggle';
    document.getElementById('toggle-enable-thinking').className = cfg.enable_thinking !== false ? 'mcp-toggle on' : 'mcp-toggle';
    toggleLLMFields();
    toggleTTSFields();
    toggleSTTFields();
    loadPolicyAudit();
    loadWeeklyMetrics();
    loadMCPServers();
  }).catch(function(e) {
    showSettingsFeedback('Load config failed: ' + (e && e.message ? e.message : 'unknown error'), true);
  });
}

// ─── Panel toggles ──────────────────────────────────────────────────

export function toggleSettings() {
  document.getElementById('settings-panel').classList.toggle('open');
}

export function toggleTTS() {
  unlockTTSAudio();  // must run inside user gesture to satisfy autoplay policy
  fetch('/api/config/tts_toggle', { method: 'POST', headers: authHeaders() }).then(function(r) { return r.json(); }).then(function(data) {
    document.getElementById('dot-tts').className = data.tts_enabled ? 'dot on' : 'dot';
    if (data.tts_status) document.getElementById('status-tts').textContent = data.tts_status;
    if (data.error) addMessage('assistant', data.error);
  });
}

export function clearChat() {
  var ws = getWs();
  if (ws) ws.send(JSON.stringify({ type: 'clear', session_key: getWebSessionKey() }));
  rotateWebSessionKey();
  clearChatState();
  document.getElementById('messages').textContent = '';
  clearRunTimelineAndRender();
}

// ─── Policy audit / metrics ─────────────────────────────────────────

export function loadPolicyAudit() {
  fetch('/api/policy/audit?limit=50', { headers: authHeaders() })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var el = document.getElementById('policy-audit-list');
      var items = (data && data.items) || [];
      if (!items.length) {
        el.textContent = '';
        var emptyDiv = document.createElement('div');
        emptyDiv.className = 'audit-item';
        emptyDiv.textContent = 'no audit records';
        el.appendChild(emptyDiv);
        return;
      }
      el.textContent = '';
      items.forEach(function(it) {
        var row = document.createElement('div');
        row.className = 'audit-item';
        var args = JSON.stringify(it.args || {});

        var statusSpan = document.createElement('span');
        statusSpan.className = 's';
        statusSpan.textContent = '[' + (it.status || '-') + ']';
        row.appendChild(statusSpan);

        row.appendChild(document.createTextNode(
          (it.tool_name || '-') + ' | ' + (it.reason || '-')
        ));
        row.appendChild(document.createElement('br'));

        var argsSpan = document.createElement('span');
        argsSpan.textContent = args;
        row.appendChild(argsSpan);

        el.appendChild(row);
      });
    })
    .catch(function() {});
}

export function loadWeeklyMetrics() {
  fetch('/api/metrics/weekly?days=7', { headers: authHeaders() })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var el = document.getElementById('weekly-metrics');
      var turns = Number(data.turns || 0);
      var success = Math.round((Number(data.task_success_rate || 0) * 1000)) / 10;
      var latency = Math.round(Number(data.avg_latency_ms || 0));
      var errRate = Math.round((Number(data.tool_error_rate || 0) * 1000)) / 10;
      var blocked = Number(data.policy_blocked_total || 0);
      var rev = Math.round((Number(data.avg_answer_revision_count || 0) * 100)) / 100;
      var planDone = Math.round((Number(data.avg_plan_completion_rate || 0) * 1000)) / 10;
      var runtimeSamples = Number(data.runtime_samples || 0);
      var queueMs = Math.round(Number(data.avg_queue_ms || 0));
      var streamMs = Math.round(Number(data.avg_stream_ms || 0));
      var ttsMs = Math.round(Number(data.avg_tts_ms || 0));
      var totalMs = Math.round(Number(data.avg_total_ms || 0));

      el.textContent = '';
      var lines = [
        ['turns:', turns],
        ['task success:', success + '%'],
        ['plan completion:', planDone + '%'],
        ['avg latency:', latency + 'ms'],
        ['tool error rate:', errRate + '%'],
        ['policy blocked:', blocked],
        ['avg revisions:', rev],
        ['runtime samples:', runtimeSamples],
        ['queue/stream/tts/total:', queueMs + '/' + streamMs + '/' + ttsMs + '/' + totalMs + 'ms'],
      ];
      lines.forEach(function(pair) {
        var div = document.createElement('div');
        var span = document.createElement('span');
        span.className = 'k';
        span.textContent = pair[0];
        div.appendChild(span);
        div.appendChild(document.createTextNode(' ' + pair[1]));
        el.appendChild(div);
      });
    })
    .catch(function() {
      var el = document.getElementById('weekly-metrics');
      el.textContent = 'failed to load metrics';
    });
}

// ─── MCP server management ──────────────────────────────────────────

export function loadMCPServers() {
  fetch('/api/config/mcp', { headers: authHeaders() })
    .then(function(r) { return r.json(); })
    .then(function(data) { renderMCPList(data.servers || []); })
    .catch(function() {});
}

export function renderMCPList(servers) {
  var list = document.getElementById('mcp-server-list');
  var addForm = list.querySelector('.mcp-add-form');
  list.textContent = '';
  if (addForm) list.appendChild(addForm);
  servers.forEach(function(s) { list.appendChild(buildMCPCard(s)); });
}

export function buildMCPCard(s) {
  var card = document.createElement('div');
  card.className = 'mcp-card' + (s.enabled ? '' : ' disabled');

  // Header
  var header = document.createElement('div');
  header.className = 'mcp-card-header';

  var nameSpan = document.createElement('span');
  nameSpan.className = 'mcp-card-name';
  nameSpan.textContent = s.name;
  header.appendChild(nameSpan);

  var actions = document.createElement('div');
  actions.className = 'mcp-card-actions';

  var toggle = document.createElement('div');
  toggle.className = 'mcp-toggle' + (s.enabled ? ' on' : '');
  toggle.addEventListener('click', function() {
    updateMCPServer(s.name, { enabled: !s.enabled });
  });
  actions.appendChild(toggle);

  var removeBtn = document.createElement('span');
  removeBtn.className = 'mcp-remove';
  removeBtn.textContent = '\u00d7';
  removeBtn.title = 'Remove';
  removeBtn.addEventListener('click', function() {
    if (confirm('Remove MCP server "' + s.name + '"?')) {
      deleteMCPServer(s.name);
    }
  });
  actions.appendChild(removeBtn);
  header.appendChild(actions);
  card.appendChild(header);

  // Command + args badge
  var cmdBadges = document.createElement('div');
  cmdBadges.className = 'mcp-badges';
  var cmdBadge = document.createElement('span');
  cmdBadge.className = 'mcp-badge';
  cmdBadge.textContent = s.command + ' ' + (s.args || []).join(' ');
  cmdBadges.appendChild(cmdBadge);
  card.appendChild(cmdBadges);

  // Risk + capability badges
  var capBadges = document.createElement('div');
  capBadges.className = 'mcp-badges';
  capBadges.style.marginTop = '4px';
  var riskBadge = document.createElement('span');
  riskBadge.className = 'mcp-badge';
  riskBadge.textContent = 'risk: ' + s.risk_level;
  capBadges.appendChild(riskBadge);
  if (s.network_access) {
    var netBadge = document.createElement('span');
    netBadge.className = 'mcp-badge';
    netBadge.textContent = 'net';
    capBadges.appendChild(netBadge);
  }
  if (s.filesystem_access) {
    var fsBadge = document.createElement('span');
    fsBadge.className = 'mcp-badge';
    fsBadge.textContent = 'fs';
    capBadges.appendChild(fsBadge);
  }
  card.appendChild(capBadges);

  // Env vars (masked)
  if (s.env_masked) {
    var envKeys = Object.keys(s.env_masked);
    if (envKeys.length > 0) {
      var envBadges = document.createElement('div');
      envBadges.className = 'mcp-badges';
      envBadges.style.marginTop = '4px';
      envKeys.forEach(function(k) {
        var b = document.createElement('span');
        b.className = 'mcp-badge';
        b.textContent = k + '=' + s.env_masked[k];
        envBadges.appendChild(b);
      });
      card.appendChild(envBadges);
    }
  }

  return card;
}

export function showAddMCPForm() {
  var MCP_PRESETS = getMcpPresets();
  var list = document.getElementById('mcp-server-list');
  if (list.querySelector('.mcp-add-form')) return;

  var form = document.createElement('div');
  form.className = 'mcp-card mcp-add-form';

  // Preset select
  var presetField = document.createElement('div');
  presetField.className = 'mcp-field';
  var presetLabel = document.createElement('label');
  presetLabel.textContent = 'PRESET';
  presetField.appendChild(presetLabel);
  var presetSelect = document.createElement('select');
  presetSelect.className = 'setting-select';
  var presetOptions = ['custom', 'filesystem', 'github', 'sqlite', 'brave-search', 'fetch'];
  presetOptions.forEach(function(p) {
    var opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p.charAt(0).toUpperCase() + p.slice(1);
    presetSelect.appendChild(opt);
  });
  presetField.appendChild(presetSelect);
  form.appendChild(presetField);

  // Name input
  var nameField = document.createElement('div');
  nameField.className = 'mcp-field';
  var nameLabel = document.createElement('label');
  nameLabel.textContent = 'NAME';
  nameField.appendChild(nameLabel);
  var nameInput = document.createElement('input');
  nameInput.className = 'setting-input';
  nameInput.placeholder = 'e.g. github';
  nameField.appendChild(nameInput);
  form.appendChild(nameField);

  // Command input
  var cmdField = document.createElement('div');
  cmdField.className = 'mcp-field';
  var cmdLabel = document.createElement('label');
  cmdLabel.textContent = 'COMMAND';
  cmdField.appendChild(cmdLabel);
  var cmdInput = document.createElement('input');
  cmdInput.className = 'setting-input';
  cmdInput.placeholder = 'e.g. npx';
  cmdField.appendChild(cmdInput);
  form.appendChild(cmdField);

  // Args input
  var argsField = document.createElement('div');
  argsField.className = 'mcp-field';
  var argsLabel = document.createElement('label');
  argsLabel.textContent = 'ARGS';
  argsField.appendChild(argsLabel);
  var argsInput = document.createElement('input');
  argsInput.className = 'setting-input';
  argsInput.placeholder = 'e.g. -y, @modelcontextprotocol/server-github';
  argsField.appendChild(argsInput);
  form.appendChild(argsField);

  // Env vars section
  var envField = document.createElement('div');
  envField.className = 'mcp-field';
  var envLabel = document.createElement('label');
  envLabel.textContent = 'ENV VARS';
  envField.appendChild(envLabel);
  var envList = document.createElement('div');
  envField.appendChild(envList);
  var addEnvBtn = document.createElement('span');
  addEnvBtn.className = 'mcp-add-env';
  addEnvBtn.textContent = '+ add env var';
  addEnvBtn.addEventListener('click', function() { addMCPEnvRow(envList); });
  envField.appendChild(addEnvBtn);
  form.appendChild(envField);

  // Risk level
  var riskField = document.createElement('div');
  riskField.className = 'mcp-field';
  var riskLabel = document.createElement('label');
  riskLabel.textContent = 'RISK LEVEL';
  riskField.appendChild(riskLabel);
  var riskSelect = document.createElement('select');
  riskSelect.className = 'setting-select';
  ['low', 'medium', 'high'].forEach(function(r) {
    var opt = document.createElement('option');
    opt.value = r;
    opt.textContent = r;
    if (r === 'medium') opt.selected = true;
    riskSelect.appendChild(opt);
  });
  riskField.appendChild(riskSelect);
  form.appendChild(riskField);

  // Checkboxes
  var checkRow = document.createElement('div');
  checkRow.className = 'mcp-inline';
  checkRow.style.marginBottom = '8px';

  var netLabel = document.createElement('label');
  var netCb = document.createElement('input');
  netCb.type = 'checkbox';
  netCb.checked = true;
  netLabel.appendChild(netCb);
  netLabel.appendChild(document.createTextNode(' Network'));
  checkRow.appendChild(netLabel);

  var fsLabel = document.createElement('label');
  var fsCb = document.createElement('input');
  fsCb.type = 'checkbox';
  fsLabel.appendChild(fsCb);
  fsLabel.appendChild(document.createTextNode(' Filesystem'));
  checkRow.appendChild(fsLabel);

  form.appendChild(checkRow);

  // Buttons
  var btnRow = document.createElement('div');
  btnRow.style.display = 'flex';
  btnRow.style.gap = '6px';

  var saveBtn = document.createElement('button');
  saveBtn.className = 'btn-apply';
  saveBtn.textContent = 'Save';
  btnRow.appendChild(saveBtn);

  var cancelBtn = document.createElement('button');
  cancelBtn.className = 'btn-apply';
  cancelBtn.style.background = 'transparent';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', function() { form.remove(); });
  btnRow.appendChild(cancelBtn);

  form.appendChild(btnRow);

  // Preset change handler
  presetSelect.addEventListener('change', function() {
    var p = MCP_PRESETS[this.value] || MCP_PRESETS.custom;
    if (this.value !== 'custom') nameInput.value = this.value;
    cmdInput.value = p.command;
    argsInput.value = (p.args || []).join(', ');
    riskSelect.value = p.risk_level || 'medium';
    netCb.checked = !!p.network_access;
    fsCb.checked = !!p.filesystem_access;
  });

  // Save handler
  saveBtn.addEventListener('click', function() {
    var name = nameInput.value.trim();
    var cmd = cmdInput.value.trim();
    if (!name || !cmd) { alert('Name and Command are required.'); return; }
    var argsRaw = argsInput.value.trim();
    var args = argsRaw ? argsRaw.split(/,\s*/) : [];
    var env = collectEnvVars(envList);
    var body = {
      name: name, command: cmd, args: args, env: env,
      risk_level: riskSelect.value,
      network_access: netCb.checked,
      filesystem_access: fsCb.checked,
      enabled: true
    };
    fetch('/api/config/mcp', {
      method: 'POST',
      headers: withAuth({'Content-Type': 'application/json'}),
      body: JSON.stringify(body)
    }).then(function(r) { return r.json(); }).then(function(d) {
      if (d.error) { alert(d.error); return; }
      renderMCPList(d.servers || []);
    });
  });

  list.insertBefore(form, list.firstChild);
}

export function addMCPEnvRow(container) {
  var row = document.createElement('div');
  row.className = 'mcp-env-row';
  var keyInput = document.createElement('input');
  keyInput.className = 'setting-input';
  keyInput.placeholder = 'KEY';
  keyInput.style.width = '40%';
  row.appendChild(keyInput);
  var valInput = document.createElement('input');
  valInput.className = 'setting-input';
  valInput.placeholder = 'value';
  valInput.style.width = '55%';
  row.appendChild(valInput);
  var removeSpan = document.createElement('span');
  removeSpan.className = 'mcp-env-remove';
  removeSpan.textContent = '\u00d7';
  removeSpan.addEventListener('click', function() { row.remove(); });
  row.appendChild(removeSpan);
  container.appendChild(row);
}

export function collectEnvVars(container) {
  var env = {};
  var rows = container.querySelectorAll('.mcp-env-row');
  for (var i = 0; i < rows.length; i++) {
    var inputs = rows[i].querySelectorAll('input');
    var k = (inputs[0].value || '').trim();
    var v = (inputs[1].value || '').trim();
    if (k) env[k] = v;
  }
  return env;
}

export function updateMCPServer(name, patch) {
  patch.name = name;
  fetch('/api/config/mcp/update', {
    method: 'POST',
    headers: withAuth({'Content-Type': 'application/json'}),
    body: JSON.stringify(patch)
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.servers) renderMCPList(d.servers);
  });
}

export function deleteMCPServer(name) {
  fetch('/api/config/mcp/delete', {
    method: 'POST',
    headers: withAuth({'Content-Type': 'application/json'}),
    body: JSON.stringify({ name: name })
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.servers) renderMCPList(d.servers);
  });
}

// ─── Tool confirmation ──────────────────────────────────────────────

export function sendToolConfirm(token, approved, buttons) {
  var ws = getWs();
  if (!ws) return;
  if (buttons && buttons.length) {
    buttons.forEach(function(b) { b.disabled = true; });
  }
  ws.send(JSON.stringify({ type: 'tool_confirm', token: token, approved: !!approved, force: false }));
}

export function sendToolConfirmWithForce(token, approved, force, buttons) {
  var ws = getWs();
  if (!ws) return;
  if (buttons && buttons.length) {
    buttons.forEach(function(b) { b.disabled = true; });
  }
  ws.send(JSON.stringify({ type: 'tool_confirm', token: token, approved: !!approved, force: !!force }));
}

// ─── Send message ───────────────────────────────────────────────────

export function sendMessage() {
  var input = document.getElementById('user-input');
  var text = input.value.trim();
  var ws = getWs();
  if (!text || !ws) return;

  unlockTTSAudio();
  resetTTSState();

  var pendImg = getPendingImage();
  var camMode = getCameraMode();

  var payload = {
    type: 'text',
    text: text,
    reuse_vision: !!camMode,
    session_key: getWebSessionKey(),
  };
  var imgUrl = null;
  if (pendImg) {
    payload.image = pendImg.base64;
    imgUrl = pendImg.dataUrl;
    clearPendingImage();
  } else if (camMode) {
    captureCameraFrame();
    var frameUrl = getCameraFrameDataUrl();
    var frameAt = getCameraFrameAtMs();
    if (frameUrl) {
      payload.image = frameUrl;
      if (frameAt > 0) payload.image_age_ms = Math.max(0, Date.now() - frameAt);
      imgUrl = frameUrl;
    }
  }

  setLastUserQuery(text);
  setLastToolUsed('');
  addMessage('user', text, imgUrl);
  ws.send(JSON.stringify(payload));
  input.value = '';
  input.style.height = 'auto';
}
