/**
 * Settings/config state store.
 * Owns: LLM provider presets, MCP presets, settings feedback timer.
 * C1: No DOM access — pure state + logic.
 *     Exception: hasStoredApiKey / currentInputValue read DOM for practicality.
 */

// ─── State ─────────────────────────────────────────────────────────
var settingsFeedbackTimer = null;

var LLM_PROVIDER_PRESET_ORDER = [
  'openai', 'gemini', 'deepseek', 'moonshot', 'openrouter', 'ollama', 'custom'
];

var LLM_PROVIDER_PRESETS = {
  openai: {
    label: 'OpenAI',
    api_base_url: 'https://api.openai.com/v1',
    api_model: 'gpt-4o',
    model_family: 'openai',
    tool_protocol: 'openai_function',
    aliases: ['api.openai.com', 'openai', 'gpt-', ' o1', ' o3', ' o4'],
  },
  gemini: {
    label: 'Gemini (OpenAI-compatible)',
    api_base_url: 'https://generativelanguage.googleapis.com/v1beta/openai/',
    api_model: 'gemini-3.0-flash',
    model_family: 'openai',
    tool_protocol: 'openai_function',
    aliases: ['generativelanguage.googleapis.com', 'gemini'],
  },
  deepseek: {
    label: 'DeepSeek',
    api_base_url: 'https://api.deepseek.com/v1',
    api_model: 'deepseek-chat',
    model_family: 'deepseek',
    tool_protocol: 'openai_function',
    aliases: ['api.deepseek.com', 'deepseek'],
  },
  moonshot: {
    label: 'Moonshot (Kimi)',
    api_base_url: 'https://api.moonshot.cn/v1',
    api_model: 'kimi-k2.5',
    model_family: 'openai',
    tool_protocol: 'openai_function',
    aliases: ['api.moonshot.cn', 'api.moonshot.ai', 'moonshot', 'moonshotai', 'kimi'],
  },
  openrouter: {
    label: 'OpenRouter',
    api_base_url: 'https://openrouter.ai/api/v1',
    api_model: 'openai/gpt-4o-mini',
    model_family: 'openai',
    tool_protocol: 'openai_function',
    aliases: ['openrouter.ai'],
  },
  ollama: {
    label: 'Ollama (local API)',
    api_base_url: 'http://127.0.0.1:11434/v1',
    api_model: 'llama3.1',
    model_family: 'llama',
    tool_protocol: 'openai_function',
    aliases: ['127.0.0.1:11434', 'localhost:11434', 'ollama'],
  },
  custom: null,
};

var MCP_PRESETS = {
  filesystem: { command: 'npx', args: ['-y', '@modelcontextprotocol/server-filesystem', '/path/to/dir'], network_access: false, filesystem_access: true, risk_level: 'medium' },
  github: { command: 'npx', args: ['-y', '@modelcontextprotocol/server-github'], network_access: true, filesystem_access: false, risk_level: 'medium' },
  sqlite: { command: 'uvx', args: ['mcp-server-sqlite', '--db-path', '/path/to/db.sqlite'], network_access: false, filesystem_access: true, risk_level: 'medium' },
  'brave-search': { command: 'npx', args: ['-y', '@modelcontextprotocol/server-brave-search'], network_access: true, filesystem_access: false, risk_level: 'low' },
  fetch: { command: 'uvx', args: ['mcp-server-fetch'], network_access: true, filesystem_access: false, risk_level: 'low' },
  custom: { command: '', args: [], network_access: true, filesystem_access: false, risk_level: 'medium' },
};

// ─── Accessors ─────────────────────────────────────────────────────
export function getSettingsFeedbackTimer() { return settingsFeedbackTimer; }
export function setSettingsFeedbackTimer(v) { settingsFeedbackTimer = v; }

export function getLlmProviderPresetOrder() { return LLM_PROVIDER_PRESET_ORDER; }
export function getLlmProviderPresets() { return LLM_PROVIDER_PRESETS; }
export function getMcpPresets() { return MCP_PRESETS; }

// ─── Pure logic ────────────────────────────────────────────────────

export function normalizeApiBaseUrl(value) {
  var v = String(value || '').trim();
  return v || 'https://api.openai.com/v1';
}

/**
 * Update LLM_PROVIDER_PRESETS from a server-supplied catalog.
 * Returns true if the presets were mutated, false otherwise.
 * The caller is responsible for calling renderLlmProviderOptions() afterward.
 */
export function setLlmProviderCatalog(catalog) {
  if (!Array.isArray(catalog) || !catalog.length) return false;
  var next = {};
  for (var i = 0; i < catalog.length; i++) {
    var item = catalog[i] || {};
    var id = String(item.id || '').trim().toLowerCase();
    if (!id || id === 'custom') continue;
    var aliases = Array.isArray(item.aliases) ? item.aliases : [];
    var cleanedAliases = [];
    for (var j = 0; j < aliases.length; j++) {
      var alias = String(aliases[j] || '').trim().toLowerCase();
      if (alias) cleanedAliases.push(alias);
    }
    if (cleanedAliases.indexOf(id) < 0) cleanedAliases.unshift(id);
    next[id] = {
      label: String(item.label || id),
      api_base_url: String(item.api_base_url || ''),
      api_model: String(item.api_model || ''),
      model_family: String(item.model_family || 'openai') || 'openai',
      tool_protocol: String(item.tool_protocol || 'openai_function') || 'openai_function',
      aliases: cleanedAliases,
    };
  }
  if (!Object.keys(next).length) return false;
  next.custom = null;
  // Clear LLM_PROVIDER_PRESETS in-place and assign new entries
  for (var k of Object.keys(LLM_PROVIDER_PRESETS)) delete LLM_PROVIDER_PRESETS[k];
  Object.assign(LLM_PROVIDER_PRESETS, next);
  return true;
}

export function inferLlmProvider(apiModel, apiUrl) {
  var model = String(apiModel || '').toLowerCase();
  var url = String(apiUrl || '').toLowerCase();
  var hay = (model + ' ' + url).trim();
  var keys = Object.keys(LLM_PROVIDER_PRESETS || {}).filter(function(k) { return k !== 'custom'; });

  function hasAlias(providerKey, source) {
    var preset = LLM_PROVIDER_PRESETS[providerKey] || {};
    var aliases = Array.isArray(preset.aliases) ? preset.aliases : [];
    if (!aliases.length) aliases = [providerKey];
    for (var i = 0; i < aliases.length; i++) {
      var alias = String(aliases[i] || '').toLowerCase();
      if (alias && source.indexOf(alias) >= 0) return true;
    }
    return false;
  }

  for (var i = 0; i < keys.length; i++) {
    var key = keys[i];
    if (key === 'openrouter' || key === 'openai' || key === 'ollama') continue;
    if (hasAlias(key, model)) return key;
  }
  for (var j = 0; j < keys.length; j++) {
    if (hasAlias(keys[j], hay)) return keys[j];
  }

  if (url.indexOf('api.openai.com') >= 0 || model.indexOf('gpt-') >= 0 || model.indexOf('o1') >= 0 || model.indexOf('o3') >= 0 || model.indexOf('o4') >= 0) return 'openai';
  return 'custom';
}

export function inferLlmFamily(apiModel, apiUrl) {
  var hay = (String(apiModel || '') + ' ' + String(apiUrl || '')).toLowerCase();
  if (hay.indexOf('deepseek') >= 0) return 'deepseek';
  if (hay.indexOf('llama') >= 0 || hay.indexOf('meta-') >= 0) return 'llama';
  if (hay.indexOf('qwen') >= 0) return 'qwen3-vl';
  var provider = inferLlmProvider(apiModel, apiUrl);
  var preset = LLM_PROVIDER_PRESETS[provider];
  if (preset && preset.model_family) return preset.model_family;
  return 'openai';
}

export function formatSTTStatus(cfg) {
  var stt = (cfg && cfg.stt) ? cfg.stt : {};
  var backend = String(stt.backend || 'local').toLowerCase();
  if (backend === 'api') {
    return 'API (' + (stt.api_model || '-') + ')';
  }
  var modelPath = String(stt.model || '');
  var modelName = modelPath ? modelPath.split('/').pop() : '-';
  return 'Local (' + modelName + ')';
}

// ─── DOM-reading utilities (pragmatic exception to C1) ─────────────

export function hasStoredApiKey(inputId) {
  var el = document.getElementById(inputId);
  if (!el) return false;
  var p = String(el.placeholder || '');
  return p.indexOf('Stored: ') === 0;
}

export function currentInputValue(inputId) {
  var el = document.getElementById(inputId);
  return String((el && el.value) || '').trim();
}

export function keyForApply(inputId, sharedKey) {
  var v = currentInputValue(inputId);
  if (v) return v;
  if (hasStoredApiKey(inputId)) return '';
  return sharedKey || '';
}
