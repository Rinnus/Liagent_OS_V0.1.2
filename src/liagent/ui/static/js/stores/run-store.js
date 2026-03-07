/**
 * Run/timeline state store.
 * Owns: runEventStore, activeRunId, timeline/filter/selection state.
 * C1: No DOM access — pure state + logic.
 * C4: Object state cleared in-place (no reference reassignment).
 */
import { sanitizeHref } from './chat-store.js';

// ─── State ─────────────────────────────────────────────────────────
const runEventStore = {};
let activeRunId = '';
let activeTimelineRunId = '';
let pinnedTimelineRunId = '';
let selectedArtifact = null;
let selectedAgent = null;
let timelineSourceFilter = '';
let timelineEventFilter = '';
let artifactKindFilter = '';
let artifactSourceFilter = '';
let artifactTextFilter = '';
const RUN_TIMELINE_LIMIT = 160;

// ─── Accessors ─────────────────────────────────────────────────────
export function getRunEventStore() { return runEventStore; }
export function getActiveRunId() { return activeRunId; }
export function setActiveRunId(id) { activeRunId = id; }
export function getActiveTimelineRunId() { return activeTimelineRunId; }
export function setActiveTimelineRunId(id) { activeTimelineRunId = id; }
export function getPinnedTimelineRunId() { return pinnedTimelineRunId; }
export function setPinnedTimelineRunId(id) { pinnedTimelineRunId = id; }
export function getSelectedArtifact() { return selectedArtifact; }
export function setSelectedArtifact(val) { selectedArtifact = val; }
export function getSelectedAgent() { return selectedAgent; }
export function setSelectedAgent(val) { selectedAgent = val; }
export function getTimelineSourceFilter() { return timelineSourceFilter; }
export function setTimelineSourceFilter(val) { timelineSourceFilter = val; }
export function getTimelineEventFilter() { return timelineEventFilter; }
export function setTimelineEventFilter(val) { timelineEventFilter = val; }
export function getArtifactKindFilter() { return artifactKindFilter; }
export function setArtifactKindFilter(val) { artifactKindFilter = val; }
export function getArtifactSourceFilter() { return artifactSourceFilter; }
export function setArtifactSourceFilter(val) { artifactSourceFilter = val; }
export function getArtifactTextFilter() { return artifactTextFilter; }
export function setArtifactTextFilter(val) { artifactTextFilter = val; }
export function getRunTimelineLimit() { return RUN_TIMELINE_LIMIT; }

// ─── Utilities ─────────────────────────────────────────────────────
export function toNumber(v, fallback) {
  var n = Number(v);
  return isFinite(n) ? n : (typeof fallback === 'number' ? fallback : 0);
}

export function shortText(text, maxLen) {
  var raw = String(text || '');
  var clean = raw.replace(/\s+/g, ' ').trim();
  var lim = Math.max(16, Number(maxLen || 120));
  if (clean.length <= lim) return clean;
  return clean.slice(0, lim - 3) + '...';
}

export function extractUrlsFromText(text) {
  var source = String(text || '');
  var re = /https?:\/\/[^\s<>"'\)\]]+/g;
  var urls = [];
  var seen = {};
  var m;
  while ((m = re.exec(source)) !== null) {
    var u = m[0];
    if (!seen[u]) {
      seen[u] = true;
      urls.push(u);
      if (urls.length >= 8) break;
    }
  }
  return urls;
}

export function formatUsd(value) {
  var n = toNumber(value, 0);
  if (n <= 0) return '$0';
  if (n < 0.01) return '$' + n.toFixed(5);
  return '$' + n.toFixed(3);
}

export function formatAgo(ms) {
  var sec = Math.max(0, Math.round((Date.now() - Number(ms || 0)) / 1000));
  if (sec < 60) return sec + 's ago';
  var min = Math.round(sec / 60);
  if (min < 60) return min + 'm ago';
  var h = Math.round(min / 60);
  return h + 'h ago';
}

function parseJSONSafe(text, fallback) {
  try { return JSON.parse(text); } catch (e) { return fallback; }
}

// ─── Timeline helpers ──────────────────────────────────────────────
function getTimelineRunId(data) {
  var id = String((data && data.run_id) || activeRunId || activeTimelineRunId || '').trim();
  return id || 'adhoc';
}

function ensureRunTimeline(runId) {
  var id = String(runId || '').trim() || 'adhoc';
  if (!runEventStore[id]) {
    runEventStore[id] = {
      events: [],
      counters: {},
      startedAt: Date.now(),
      lastEventAt: Date.now(),
      lastState: 'accepted',
      query: '',
      agents: {},
      routing: {
        mode: '-',
        workerCount: 0,
        apiTokenPressure: '-',
        apiRecentTokens: 0,
        apiCostPressure: '-',
        apiRecentCostUsd: 0,
        serviceTier: '-',
        skill: '',
        adapted: false,
        adaptReason: '',
        reasonCodes: [],
        classification: {},
        pressure: {},
        backend: '-',
        modelRole: '-',
        toolBackend: '-',
        ragTier: '-',
        taskClass: '-',
        dataSensitivity: '-',
        decisionNotes: [],
        updates: 0,
      },
      usage: {
        provider: '-',
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        cachedPromptTokens: 0,
        estimatedCostUsd: 0,
        usageEvents: 0,
        cacheHitRatio: 0,
        turnBudget: null,
        turnUsed: null,
      },
      artifacts: [],
      artifactFingerprints: {},
    };
  }
  return runEventStore[id];
}

function buildTimelineDetail(data) {
  if (!data || typeof data !== 'object') return '';
  if (data.type === 'tool_start') return (data.name || '-') + ' start';
  if (data.type === 'tool_result') return (data.name || '-') + ' done';
  if (data.type === 'error') return String(data.text || 'error').slice(0, 80);
  if (data.type === 'sub_complete') {
    if (data.data && typeof data.data === 'object') {
      var ok = data.data.success === false ? 'fail' : 'ok';
      var summary = String(data.data.summary || data.data.error || '').slice(0, 64);
      var score = Number(data.data.score || 0).toFixed(2);
      return ok + ' score=' + score + ' ' + summary;
    }
    return String(data.data || '').slice(0, 80);
  }
  if (data.type === 'synthesis_partial') {
    var partial = (data.data && typeof data.data === 'object') ? data.data : {};
    return 'partial synthesis agents=' + Number(partial.agent_count || 0);
  }
  if (data.type === 'dispatch') {
    var mode = (data.data && data.data.mode) ? data.data.mode : '';
    var workerCount = Number((data.data && data.data.worker_count) || 0);
    return (mode || '-') + ' workers=' + workerCount;
  }
  if (data.type === 'run_state') return String(data.state || '');
  if (data.type === 'llm_usage') {
    var usage = (data && data.usage && typeof data.usage === 'object') ? data.usage : {};
    return 'tokens=' + Number(usage.total_tokens || 0);
  }
  if (data.type === 'done') return 'completed';
  return '';
}

function addArtifact(bucket, artifact) {
  if (!bucket || !artifact) return;
  var kind = String(artifact.kind || 'artifact');
  var source = String(artifact.source || '-');
  var text = String(artifact.text || '').trim();
  var url = String(artifact.url || '').trim();
  if (!text && !url) return;
  var fp = kind + '|' + source + '|' + text + '|' + url;
  if (bucket.artifactFingerprints[fp]) return;
  bucket.artifactFingerprints[fp] = 1;
  bucket.artifacts.unshift({
    id: Date.now().toString(36) + Math.random().toString(16).slice(2, 8),
    kind: kind,
    source: source,
    text: shortText(text || url, 180),
    rawText: String(text || url || ''),
    url: url,
    ts: Date.now(),
  });
  if (bucket.artifacts.length > 80) {
    bucket.artifacts.length = 80;
  }
}

function touchAgent(bucket, key) {
  var k = String(key || '').trim();
  if (!k) return null;
  if (!bucket.agents[k]) {
    bucket.agents[k] = {
      id: k,
      events: 0,
      toolStarts: 0,
      toolResults: 0,
      completed: 0,
      errors: 0,
      lastEvent: '-',
      lastDetail: '',
      lastTs: 0,
      strategy: '',
      summary: '',
      sourceUrls: [],
      sourceCount: 0,
      llmCalls: 0,
      score: 0,
      status: 'running',
      error: '',
    };
  }
  return bucket.agents[k];
}

function updateRunAggregates(bucket, data, eventType, entry) {
  if (!bucket || !data) return;
  var src = String(data.event_source || data.agent_id || '').trim();
  var agent = touchAgent(bucket, src);
  if (agent) {
    agent.events += 1;
    if (eventType === 'tool_start') agent.toolStarts += 1;
    if (eventType === 'tool_result') agent.toolResults += 1;
    if (eventType === 'sub_complete' || data.type === 'sub_complete') agent.completed += 1;
    if (eventType === 'error' || data.type === 'error') agent.errors += 1;
    agent.lastEvent = eventType || data.type || '-';
    agent.lastDetail = shortText(entry.detail || '', 64);
    agent.lastTs = Number(entry.ts || 0);
    if (data.type === 'sub_complete' && data.data && typeof data.data === 'object') {
      var sub = data.data;
      if (sub.strategy) agent.strategy = String(sub.strategy);
      if (sub.summary) agent.summary = String(sub.summary);
      if (Array.isArray(sub.source_urls)) {
        agent.sourceUrls = sub.source_urls.slice(0, 12).map(function(u) { return String(u || ''); });
        agent.sourceCount = agent.sourceUrls.length;
      } else if (typeof sub.source_count === 'number') {
        agent.sourceCount = Number(sub.source_count || 0);
      }
      if (typeof sub.llm_calls === 'number') agent.llmCalls = Number(sub.llm_calls || 0);
      if (typeof sub.score === 'number') agent.score = Math.max(0, Math.min(1, Number(sub.score)));
      agent.status = sub.success === false ? 'failed' : 'done';
      agent.error = String(sub.error || '');
    }
  }

  if (data.type === 'dispatch') {
    var dispatchData = (data.data && typeof data.data === 'object') ? data.data : {};
    if (dispatchData.query) bucket.query = shortText(dispatchData.query, 220);
    var r = bucket.routing || {};
    var dispatchPressure = (dispatchData.pressure && typeof dispatchData.pressure === 'object')
      ? dispatchData.pressure
      : {};
    r.mode = String(dispatchData.mode || r.mode || '-');
    r.workerCount = toNumber(dispatchData.worker_count, r.workerCount || 0);
    r.apiTokenPressure = String(
      dispatchPressure.token_label || dispatchData.api_token_pressure_level || r.apiTokenPressure || '-'
    );
    r.apiRecentTokens = toNumber(dispatchData.api_recent_tokens, r.apiRecentTokens || 0);
    r.apiCostPressure = String(
      dispatchPressure.cost_label || dispatchData.api_cost_pressure_level || r.apiCostPressure || '-'
    );
    r.apiRecentCostUsd = toNumber(dispatchData.api_recent_cost_usd, r.apiRecentCostUsd || 0);
    r.reasonCodes = Array.isArray(dispatchData.reason_codes) ? dispatchData.reason_codes.slice(0, 12) : (r.reasonCodes || []);
    r.classification = (dispatchData.classification && typeof dispatchData.classification === 'object')
      ? dispatchData.classification
      : (r.classification || {});
    r.pressure = (dispatchData.pressure && typeof dispatchData.pressure === 'object')
      ? dispatchData.pressure
      : (r.pressure || {});
    var routingDecision = (dispatchData.routing_decision && typeof dispatchData.routing_decision === 'object')
      ? dispatchData.routing_decision
      : {};
    r.backend = String(routingDecision.backend || r.backend || '-');
    r.modelRole = String(routingDecision.model_role || r.modelRole || '-');
    r.toolBackend = String(routingDecision.tool_backend || r.toolBackend || '-');
    r.ragTier = String(routingDecision.rag_tier || r.ragTier || '-');
    r.taskClass = String(routingDecision.task_class || r.taskClass || '-');
    r.dataSensitivity = String(routingDecision.data_sensitivity || r.dataSensitivity || '-');
    if (!Array.isArray(r.decisionNotes)) r.decisionNotes = [];
    r.decisionNotes.unshift('dispatch mode=' + r.mode + ' workers=' + r.workerCount);
    if (r.decisionNotes.length > 8) r.decisionNotes.length = 8;
    r.updates = toNumber(r.updates, 0) + 1;
    bucket.routing = r;
  }

  if (data.type === 'service_tier') {
    var tierObj = (typeof data.result === 'string') ? parseJSONSafe(data.result, {}) : (data.result || {});
    if (tierObj && typeof tierObj === 'object') {
      var rt = bucket.routing || {};
      rt.serviceTier = String(tierObj.tier || rt.serviceTier || '-');
      rt.skill = String(tierObj.skill || rt.skill || '');
      rt.adapted = !!tierObj.adapted;
      rt.adaptReason = String(tierObj.adapt_reason || rt.adaptReason || '');
      if (!Array.isArray(rt.decisionNotes)) rt.decisionNotes = [];
      rt.decisionNotes.unshift(
        'tier=' + (rt.serviceTier || '-') +
        (rt.adapted ? (' adapted(' + (rt.adaptReason || 'overload') + ')') : ' stable')
      );
      if (rt.decisionNotes.length > 8) rt.decisionNotes.length = 8;
      rt.updates = toNumber(rt.updates, 0) + 1;
      bucket.routing = rt;
    }
  }

  if (data.type === 'skill_selected') {
    var skillObj = (typeof data.result === 'string') ? parseJSONSafe(data.result, {}) : (data.result || {});
    if (skillObj && typeof skillObj === 'object') {
      var rs = bucket.routing || {};
      if (skillObj.name) rs.skill = String(skillObj.name);
      if (!Array.isArray(rs.decisionNotes)) rs.decisionNotes = [];
      if (skillObj.name) {
        rs.decisionNotes.unshift(
          'skill=' + String(skillObj.name) +
          (typeof skillObj.confidence === 'number' ? (' conf=' + Number(skillObj.confidence).toFixed(2)) : '')
        );
      }
      if (rs.decisionNotes.length > 8) rs.decisionNotes.length = 8;
      rs.updates = toNumber(rs.updates, 0) + 1;
      bucket.routing = rs;
    }
  }

  if (data.type === 'llm_usage') {
    var usage = (data.usage && typeof data.usage === 'object') ? data.usage : {};
    var u = bucket.usage;
    if (usage.provider) u.provider = String(usage.provider);
    u.promptTokens += toNumber(usage.prompt_tokens, 0);
    u.completionTokens += toNumber(usage.completion_tokens, 0);
    u.totalTokens += toNumber(usage.total_tokens, 0);
    u.cachedPromptTokens += toNumber(usage.cached_prompt_tokens, 0);
    u.estimatedCostUsd += Math.max(0, toNumber(usage.estimated_cost_usd, 0));
    u.usageEvents += 1;
    if (typeof usage.turn_budget === 'number') u.turnBudget = Number(usage.turn_budget);
    if (typeof usage.turn_used === 'number') u.turnUsed = Number(usage.turn_used);
    if (typeof usage.cache_hit_ratio === 'number') {
      u.cacheHitRatio = Math.max(0, Math.min(1, Number(usage.cache_hit_ratio)));
    } else if (u.promptTokens > 0) {
      u.cacheHitRatio = Math.max(0, Math.min(1, u.cachedPromptTokens / u.promptTokens));
    }
  }

  if (data.type === 'tool_result') {
    var toolName = String(data.name || 'tool');
    var resultText = String(data.result || '');
    addArtifact(bucket, {
      kind: 'tool_result',
      source: toolName,
      text: resultText,
      url: '',
    });
    var urls = extractUrlsFromText(resultText);
    for (var i = 0; i < urls.length; i++) {
      addArtifact(bucket, {
        kind: 'evidence_url',
        source: toolName,
        text: urls[i],
        url: urls[i],
      });
    }
  }

  if (data.type === 'synthesis') {
    var syn = (data.data && typeof data.data === 'object') ? data.data : {};
    if (syn.agent_scores && typeof syn.agent_scores === 'object') {
      Object.keys(syn.agent_scores).forEach(function(agentId) {
        var ag = touchAgent(bucket, agentId);
        if (!ag) return;
        var val = Number(syn.agent_scores[agentId] || 0);
        if (!isNaN(val)) ag.score = Math.max(0, Math.min(1, val));
      });
    }
    if (syn.answer) {
      addArtifact(bucket, {
        kind: 'synthesis',
        source: 'lead',
        text: syn.answer,
        url: '',
      });
    }
    var citations = [];
    if (Array.isArray(syn.citations)) {
      citations = syn.citations.slice(0, 40);
    } else if (syn.citations && typeof syn.citations === 'object') {
      Object.keys(syn.citations).slice(0, 20).forEach(function(key) {
        var v = syn.citations[key];
        if (Array.isArray(v)) {
          v.slice(0, 6).forEach(function(item) {
            citations.push({ text: key + ': ' + String(item || ''), url: String(item || '') });
          });
        } else {
          citations.push({ text: key + ': ' + String(v || ''), url: String(v || '') });
        }
      });
    }
    for (var j = 0; j < citations.length; j++) {
      var c = citations[j];
      var text = '';
      var url = '';
      if (typeof c === 'string') {
        text = c;
      } else if (c && typeof c === 'object') {
        text = String(c.title || c.text || c.url || JSON.stringify(c));
        url = String(c.url || '');
      }
      if (url && !sanitizeHref(url)) url = '';
      addArtifact(bucket, {
        kind: 'citation',
        source: 'lead',
        text: text,
        url: url,
      });
    }
  }

  if (data.type === 'synthesis_partial') {
    var ps = (data.data && typeof data.data === 'object') ? data.data : {};
    if (ps.answer) {
      addArtifact(bucket, {
        kind: 'synthesis_partial',
        source: 'lead',
        text: ps.answer,
        url: '',
      });
    }
  }

  if (data.type === 'context_update') {
    var ctxObj = (typeof data.data === 'string') ? parseJSONSafe(data.data, {}) : (data.data || {});
    if (ctxObj && typeof ctxObj === 'object' && (ctxObj.preview || ctxObj.var)) {
      addArtifact(bucket, {
        kind: 'context',
        source: String(ctxObj.tool || 'context'),
        text: String(ctxObj.preview || ('$' + (ctxObj.var || '?'))),
        url: '',
      });
    }
  }
}

// ─── Actions ───────────────────────────────────────────────────────

/**
 * Track a WS event into the run store.
 * Returns { runId, shouldRender } — caller is responsible for triggering
 * the appropriate renderer (C1: store does not touch DOM).
 */
export function trackRunEvent(data) {
  if (!data || typeof data !== 'object') return { runId: '', shouldRender: false };
  var hasEnvelope = !!data.event_type || !!data.event_source || typeof data.event_seq === 'number';
  var isState = data.type === 'run_state';
  if (!hasEnvelope && !isState) return { runId: '', shouldRender: false };
  var runId = getTimelineRunId(data);
  var row = ensureRunTimeline(runId);
  row.lastEventAt = Date.now();
  var eventType = String(data.event_type || data.type || 'event');
  var entry = {
    phase: String(data.type || 'event'),
    eventType: eventType,
    source: String(data.event_source || '-'),
    agent: String(data.agent_id || '-'),
    seq: (typeof data.event_seq === 'number') ? data.event_seq : null,
    ts: (typeof data.event_ts === 'number') ? data.event_ts : (Date.now() / 1000.0),
    detail: buildTimelineDetail(data),
  };
  row.events.push(entry);
  if (row.events.length > RUN_TIMELINE_LIMIT) row.events.shift();
  row.counters[eventType] = Number(row.counters[eventType] || 0) + 1;
  if (isState) row.lastState = String(data.state || row.lastState || 'idle');
  updateRunAggregates(row, data, eventType, entry);
  if (!activeTimelineRunId) activeTimelineRunId = runId;
  var shouldRender = !pinnedTimelineRunId || pinnedTimelineRunId === runId;
  if (shouldRender) activeTimelineRunId = runId;
  return { runId: runId, shouldRender: shouldRender };
}

export function latestRunId() {
  var ids = Object.keys(runEventStore);
  if (!ids.length) return '';
  ids.sort(function(a, b) {
    var x = toNumber((runEventStore[a] || {}).lastEventAt, 0);
    var y = toNumber((runEventStore[b] || {}).lastEventAt, 0);
    return y - x;
  });
  return ids[0] || '';
}

/**
 * Clear all run data in-place (C4: no reference reassignment).
 */
export function clearRunTimeline() {
  for (var key of Object.keys(runEventStore)) delete runEventStore[key];
  activeTimelineRunId = '';
  pinnedTimelineRunId = '';
  selectedArtifact = null;
  selectedAgent = null;
  timelineSourceFilter = '';
  timelineEventFilter = '';
  artifactKindFilter = '';
  artifactSourceFilter = '';
  artifactTextFilter = '';
}
