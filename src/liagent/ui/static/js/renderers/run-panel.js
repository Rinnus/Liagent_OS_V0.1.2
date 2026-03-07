/**
 * Run panel renderer — timeline, agents, artifacts, cost, routing.
 * C1: Reads stores via getters, writes to DOM only.
 * C6: scheduleRenderRunTimeline uses requestAnimationFrame throttle.
 */
import {
  getRunEventStore, getActiveTimelineRunId, setActiveTimelineRunId,
  getPinnedTimelineRunId, setPinnedTimelineRunId,
  getSelectedArtifact, setSelectedArtifact,
  getSelectedAgent, setSelectedAgent,
  getTimelineSourceFilter, setTimelineSourceFilter,
  getTimelineEventFilter, setTimelineEventFilter,
  getArtifactKindFilter, setArtifactKindFilter,
  getArtifactSourceFilter, setArtifactSourceFilter,
  getArtifactTextFilter,
  latestRunId, clearRunTimeline as clearRunStore,
  toNumber, shortText, formatUsd, formatAgo,
} from '../stores/run-store.js';
import { sanitizeHref } from '../stores/chat-store.js';

// ─── Utility ───────────────────────────────────────────────────────
function escHtml(s) {
  var d = document.createElement('div');
  d.appendChild(document.createTextNode(s));
  return d.innerHTML;
}

// ─── Render throttle (C6) ──────────────────────────────────────────
var _renderRAF = 0;
var _pendingRunId = '';

export function scheduleRenderRunTimeline(runId) {
  _pendingRunId = runId;
  if (!_renderRAF) {
    _renderRAF = requestAnimationFrame(function() {
      _renderRAF = 0;
      renderRunTimeline(_pendingRunId);
    });
  }
}

// ─── Filter helpers ────────────────────────────────────────────────
function setSelectOptions(selectEl, options, currentValue, allLabel) {
  if (!selectEl) return '';
  var current = String(currentValue || '').trim();
  var uniq = [];
  var seen = {};
  for (var i = 0; i < options.length; i++) {
    var v = String(options[i] || '').trim();
    if (!v || seen[v]) continue;
    seen[v] = true;
    uniq.push(v);
  }
  uniq.sort();
  selectEl.textContent = '';
  var allOpt = document.createElement('option');
  allOpt.value = '';
  allOpt.textContent = allLabel || 'all';
  selectEl.appendChild(allOpt);
  for (var j = 0; j < uniq.length; j++) {
    var opt = document.createElement('option');
    opt.value = uniq[j];
    opt.textContent = uniq[j];
    selectEl.appendChild(opt);
  }
  var valid = current && seen[current];
  selectEl.value = valid ? current : '';
  return valid ? current : '';
}

function rebuildTimelineFilterOptions(bucket) {
  var sourceSelect = document.getElementById('timeline-source-filter');
  var eventSelect = document.getElementById('timeline-event-filter');
  if (!sourceSelect || !eventSelect) return;
  var events = (bucket && Array.isArray(bucket.events)) ? bucket.events : [];
  var sources = [];
  var types = [];
  for (var i = 0; i < events.length; i++) {
    var e = events[i] || {};
    var src = String(e.source || e.agent || '').trim();
    var typ = String(e.eventType || '').trim();
    if (src && src !== '-') sources.push(src);
    if (typ) types.push(typ);
  }
  setTimelineSourceFilter(setSelectOptions(sourceSelect, sources, getTimelineSourceFilter(), 'source: all'));
  setTimelineEventFilter(setSelectOptions(eventSelect, types, getTimelineEventFilter(), 'event: all'));
}

function rebuildArtifactFilterOptions(bucket) {
  var kindSelect = document.getElementById('artifact-kind-filter');
  var sourceSelect = document.getElementById('artifact-source-filter');
  if (!kindSelect || !sourceSelect) return;
  var items = (bucket && Array.isArray(bucket.artifacts)) ? bucket.artifacts : [];
  var kinds = [];
  var sources = [];
  for (var i = 0; i < items.length; i++) {
    var it = items[i] || {};
    if (it.kind) kinds.push(String(it.kind));
    if (it.source) sources.push(String(it.source));
  }
  setArtifactKindFilter(setSelectOptions(kindSelect, kinds, getArtifactKindFilter(), 'kind: all'));
  setArtifactSourceFilter(setSelectOptions(sourceSelect, sources, getArtifactSourceFilter(), 'source: all'));
}

function applyTimelineFilters(events) {
  var input = Array.isArray(events) ? events : [];
  var out = [];
  var srcFilter = getTimelineSourceFilter();
  var evtFilter = getTimelineEventFilter();
  for (var i = 0; i < input.length; i++) {
    var e = input[i] || {};
    if (srcFilter) {
      var src = String(e.source || e.agent || '').trim();
      if (src !== srcFilter) continue;
    }
    if (evtFilter) {
      var et = String(e.eventType || '').trim();
      if (et !== evtFilter) continue;
    }
    out.push(e);
  }
  return out;
}

function applyArtifactFilters(items) {
  var input = Array.isArray(items) ? items : [];
  var out = [];
  var kindF = getArtifactKindFilter();
  var sourceF = getArtifactSourceFilter();
  var textF = getArtifactTextFilter();
  for (var i = 0; i < input.length; i++) {
    var it = input[i] || {};
    if (kindF && String(it.kind || '') !== kindF) continue;
    if (sourceF && String(it.source || '') !== sourceF) continue;
    if (textF) {
      var hay = (String(it.text || '') + ' ' + String(it.rawText || '') + ' ' + String(it.url || '')).toLowerCase();
      if (hay.indexOf(textF) < 0) continue;
    }
    out.push(it);
  }
  return out;
}

function updateFilterMeta(metaId, shown, total, noun) {
  var el = document.getElementById(metaId);
  if (!el) return;
  var n = noun || 'items';
  if (shown === total) {
    el.textContent = 'showing all ' + total + ' ' + n;
    return;
  }
  el.textContent = 'showing ' + shown + '/' + total + ' ' + n;
}

// ─── Selection ─────────────────────────────────────────────────────
export function setTimelineLiveMode(enabled) {
  var liveBtn = document.getElementById('run-panel-live');
  if (enabled) {
    setPinnedTimelineRunId('');
    if (liveBtn) liveBtn.classList.add('on');
    var latestId = latestRunId();
    if (latestId) {
      setActiveTimelineRunId(latestId);
      renderRunTimeline(latestId);
    } else {
      setActiveTimelineRunId('');
      renderRunTimeline('');
    }
    return;
  }
  if (liveBtn) liveBtn.classList.remove('on');
}

export function selectTimelineRun(runId) {
  var id = String(runId || '').trim();
  if (!id || !getRunEventStore()[id]) return;
  setPinnedTimelineRunId(id);
  setTimelineLiveMode(false);
  setActiveTimelineRunId(id);
  clearArtifactSelection();
  clearAgentSelection();
  renderRunTimeline(id);
}

export function clearAgentSelection() {
  setSelectedAgent(null);
  var rid = getActiveTimelineRunId();
  var bucket = rid ? getRunEventStore()[rid] : null;
  renderAgentDrawer(bucket, rid || '');
  if (bucket) renderAgentBoard(bucket);
}

export function selectAgent(runId, agentId) {
  var rid = String(runId || '').trim();
  var aid = String(agentId || '').trim();
  if (!rid || !aid) return;
  setSelectedAgent({ runId: rid, agentId: aid });
  var bucket = getRunEventStore()[rid] || null;
  renderAgentBoard(bucket);
  renderAgentDrawer(bucket, rid);
}

export function clearArtifactSelection() {
  setSelectedArtifact(null);
  var rid = getActiveTimelineRunId();
  var bucket = rid ? getRunEventStore()[rid] : null;
  renderArtifactDrawer(bucket, rid || '');
  if (bucket) renderArtifacts(bucket);
}

export function selectArtifact(runId, artifactId) {
  var rid = String(runId || '').trim();
  var aid = String(artifactId || '').trim();
  if (!rid || !aid) return;
  setSelectedArtifact({ runId: rid, artifactId: aid });
  var bucket = getRunEventStore()[rid] || null;
  renderArtifacts(bucket);
  renderArtifactDrawer(bucket, rid);
}

export function exportArtifacts() {
  var runId = String(getActiveTimelineRunId() || '').trim();
  var bucket = runId ? getRunEventStore()[runId] : null;
  if (!bucket) return;
  var items = applyArtifactFilters(Array.isArray(bucket.artifacts) ? bucket.artifacts : []);
  var payload = {
    run_id: runId,
    exported_at: new Date().toISOString(),
    query: bucket.query || '',
    routing: bucket.routing || {},
    usage: bucket.usage || {},
    filters: {
      kind: getArtifactKindFilter(),
      source: getArtifactSourceFilter(),
      text: getArtifactTextFilter(),
    },
    count: items.length,
    artifacts: items,
  };
  var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8' });
  var href = URL.createObjectURL(blob);
  var a = document.createElement('a');
  var stamp = new Date().toISOString().replace(/[:\.]/g, '-');
  a.href = href;
  a.download = 'liagent-artifacts-' + runId.slice(0, 6) + '-' + stamp + '.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(function() { URL.revokeObjectURL(href); }, 1200);
}

// ─── Renderers ─────────────────────────────────────────────────────
export function renderAgentBoard(bucket) {
  var listEl = document.getElementById('run-agent-list');
  var emptyEl = document.getElementById('run-agent-empty');
  var metaEl = document.getElementById('agent-filter-meta');
  if (!listEl || !emptyEl) return;
  if (!bucket) {
    listEl.textContent = '';
    emptyEl.style.display = '';
    emptyEl.textContent = 'no agent events yet';
    if (metaEl) metaEl.textContent = 'latest sub-agent snapshots';
    renderAgentDrawer(null, '');
    return;
  }
  var ids = Object.keys(bucket.agents || {});
  ids.sort(function(a, b) {
    var x = (bucket.agents[a] && bucket.agents[a].lastTs) || 0;
    var y = (bucket.agents[b] && bucket.agents[b].lastTs) || 0;
    return y - x;
  });
  listEl.textContent = '';
  var shown = 0;
  var sel = getSelectedAgent();
  var activeAgentId = (sel && sel.runId === getActiveTimelineRunId())
    ? String(sel.agentId || '')
    : '';
  for (var i = 0; i < ids.length && shown < 10; i++) {
    var agent = bucket.agents[ids[i]];
    if (!agent) continue;
    var row = document.createElement('div');
    row.className = 'agent-item' + (activeAgentId && String(agent.id || '') === activeAgentId ? ' active' : '');
    var scoreText = Number(agent.score || 0).toFixed(2);
    var stateText = String(agent.status || (agent.errors > 0 ? 'failed' : '-'));
    var sourceText = Number(agent.sourceCount || 0);
    row.innerHTML =
      '<div class="h"><span>' + escHtml(agent.id) + '</span><span class="m">' + escHtml(agent.lastEvent) + '</span></div>' +
      '<div class="s">state=' + escHtml(stateText) + ' | src=' + sourceText + ' | <span class="score">score=' + scoreText + '</span></div>' +
      '<div class="s">evt=' + agent.events + ' | tool=' + agent.toolStarts + '/' + agent.toolResults +
      ' | done=' + agent.completed + ' | err=' + agent.errors + '</div>' +
      (agent.error ? ('<div class="err">' + escHtml(shortText(String(agent.error), 80)) + '</div>') : '') +
      (agent.lastDetail ? ('<div class="m">' + escHtml(agent.lastDetail) + '</div>') : '');
    (function(runId, agentId) {
      row.addEventListener('click', function() { selectAgent(runId, agentId); });
    })(getActiveTimelineRunId() || '', String(agent.id || ''));
    listEl.appendChild(row);
    shown += 1;
  }
  emptyEl.style.display = shown ? 'none' : '';
  if (!shown) emptyEl.textContent = 'no agent events yet';
  if (metaEl) metaEl.textContent = shown ? ('showing ' + shown + ' agents') : 'latest sub-agent snapshots';
  renderAgentDrawer(bucket, getActiveTimelineRunId() || '');
}

export function renderRoutingTrace(bucket) {
  var bodyEl = document.getElementById('run-route-body');
  if (!bodyEl) return;
  if (!bucket) {
    bodyEl.textContent = 'waiting for route decisions...';
    return;
  }
  var r = bucket.routing || {};
  var route = String(r.mode || '-');
  var workerCount = toNumber(r.workerCount, 0);
  var tokenPressure = String(r.apiTokenPressure || '-');
  var recentTokens = toNumber(r.apiRecentTokens, 0);
  var costPressure = String(r.apiCostPressure || '-');
  var recentCost = toNumber(r.apiRecentCostUsd, 0);
  var tier = String(r.serviceTier || '-');
  var skill = String(r.skill || '');
  var adapt = r.adapted ? ('adapted(' + (r.adaptReason || 'overload') + ')') : 'stable';
  var cls = (r.classification && typeof r.classification === 'object') ? r.classification : {};
  var p = (r.pressure && typeof r.pressure === 'object') ? r.pressure : {};
  var reasonCodes = Array.isArray(r.reasonCodes) ? r.reasonCodes.slice(0, 6) : [];
  var backend = String(r.backend || '-');
  var modelRole = String(r.modelRole || '-');
  var toolBackend = String(r.toolBackend || '-');
  var ragTier = String(r.ragTier || '-');
  var taskClass = String(r.taskClass || '-');
  var dataSensitivity = String(r.dataSensitivity || '-');
  var entityCount = toNumber(cls.entity_count, 0);
  var researchSignal = !!cls.has_research_signal;
  var deepSignal = !!cls.deep_signal;
  var tokenSoft = toNumber(p.token_soft_threshold, 0);
  var tokenHard = toNumber(p.token_hard_threshold, 0);
  var costSoft = toNumber(p.cost_soft_threshold, 0);
  var costHard = toNumber(p.cost_hard_threshold, 0);
  var tokenLabel = String(p.token_label || '-');
  var costLabel = String(p.cost_label || '-');
  var explain = [];
  if (route === 'research') explain.push('multi-agent fan-out enabled');
  if (route === 'passthrough') explain.push('single-agent path');
  if (workerCount > 1) explain.push('parallel workers=' + workerCount);
  if (tokenPressure === 'hard' || costPressure === 'hard') explain.push('aggressive budget pressure');
  else if (tokenPressure === 'soft' || costPressure === 'soft') explain.push('soft budget pressure');
  if (r.adapted) explain.push('tier adapted for reliability/cost');
  if (skill) explain.push('skill route active');
  var noteList = Array.isArray(r.decisionNotes) ? r.decisionNotes.slice(0, 3) : [];
  var noteText = noteList.length ? noteList.join(' | ') : '-';
  bodyEl.innerHTML =
    '<div><span class="k">route:</span> ' + escHtml(route) + ' | workers=' + workerCount + '</div>' +
    '<div><span class="k">decision:</span> backend=' + escHtml(backend) +
    ' | role=' + escHtml(modelRole) +
    ' | tool=' + escHtml(toolBackend) +
    ' | rag=' + escHtml(ragTier) + '</div>' +
    '<div><span class="k">task:</span> ' + escHtml(taskClass) +
    ' | sensitivity=' + escHtml(dataSensitivity) + '</div>' +
    '<div><span class="k">token pressure:</span> ' + escHtml(tokenPressure) +
    ' | recent=' + recentTokens + '</div>' +
    '<div><span class="k">cost pressure:</span> ' + escHtml(costPressure) +
    ' | recent=' + formatUsd(recentCost) + '</div>' +
    '<div><span class="k">thresholds:</span> token ' + tokenSoft + '/' + tokenHard +
    ' | cost ' + formatUsd(costSoft) + '/' + formatUsd(costHard) + '</div>' +
    '<div><span class="k">signals:</span> entity=' + entityCount +
    ' | research=' + (researchSignal ? 'yes' : 'no') +
    ' | deep=' + (deepSignal ? 'yes' : 'no') +
    ' | t/c=' + escHtml(tokenLabel + '/' + costLabel) + '</div>' +
    '<div><span class="k">tier:</span> ' + escHtml(tier) +
    (skill ? (' | skill=' + escHtml(skill)) : '') + '</div>' +
    '<div><span class="k">status:</span> ' + escHtml(adapt) + '</div>' +
    '<div><span class="k">explain:</span> ' + escHtml(explain.join(' | ') || 'baseline routing') + '</div>' +
    '<div><span class="k">reasons:</span> ' + escHtml(reasonCodes.join(' | ') || '-') + '</div>' +
    '<div><span class="k">notes:</span> ' + escHtml(noteText) + '</div>';
}

export function renderRunHistory() {
  var listEl = document.getElementById('run-history-list');
  var emptyEl = document.getElementById('run-history-empty');
  if (!listEl || !emptyEl) return;
  var store = getRunEventStore();
  var ids = Object.keys(store);
  ids.sort(function(a, b) {
    var x = toNumber((store[a] || {}).lastEventAt, 0);
    var y = toNumber((store[b] || {}).lastEventAt, 0);
    return y - x;
  });
  listEl.textContent = '';
  var shown = 0;
  var activeId = getActiveTimelineRunId();
  for (var i = 0; i < ids.length && shown < 12; i++) {
    var id = ids[i];
    var bucket = store[id];
    if (!bucket) continue;
    var item = document.createElement('div');
    item.className = 'run-history-item' + (id === activeId ? ' active' : '');
    var modeText = (bucket.routing && bucket.routing.mode) ? bucket.routing.mode : '-';
    var stateText = bucket.lastState || 'accepted';
    item.innerHTML =
      '<div class="h"><span>#' + escHtml(id.slice(0, 6)) + '</span><span class="m">' +
      escHtml(stateText) + '</span></div>' +
      '<div class="m">mode=' + escHtml(modeText) + ' | ' +
      (bucket.events ? bucket.events.length : 0) + ' events | ' +
      formatAgo(bucket.lastEventAt || bucket.startedAt) + '</div>';
    (function(runId) {
      item.addEventListener('click', function() { selectTimelineRun(runId); });
    })(id);
    listEl.appendChild(item);
    shown += 1;
  }
  emptyEl.style.display = shown ? 'none' : '';
  if (!shown) emptyEl.textContent = 'no runs yet';
}

export function renderAgentDrawer(bucket, runId) {
  var drawerEl = document.getElementById('run-agent-drawer');
  var titleEl = document.getElementById('run-agent-title');
  var bodyEl = document.getElementById('run-agent-body');
  if (!drawerEl || !titleEl || !bodyEl) return;
  var sel = getSelectedAgent();
  if (!bucket || !sel || sel.runId !== runId) {
    drawerEl.classList.remove('open');
    titleEl.textContent = 'agent';
    bodyEl.textContent = '';
    return;
  }
  var aid = String(sel.agentId || '');
  var agent = (bucket.agents && bucket.agents[aid]) ? bucket.agents[aid] : null;
  if (!agent) {
    drawerEl.classList.remove('open');
    titleEl.textContent = 'agent';
    bodyEl.textContent = '';
    return;
  }
  titleEl.textContent = aid + ' | ' + (agent.lastEvent || '-');
  var urls = Array.isArray(agent.sourceUrls) ? agent.sourceUrls : [];
  var summary = String(agent.summary || '').trim();
  if (summary.length > 6000) summary = summary.slice(0, 6000) + '\n...[truncated]...';
  var bodyLines = [];
  bodyLines.push('strategy: ' + String(agent.strategy || '-'));
  bodyLines.push('status: ' + String(agent.status || '-'));
  bodyLines.push('score: ' + Number(agent.score || 0).toFixed(2));
  bodyLines.push('events/tool/done/err: ' + Number(agent.events || 0) + '/' + Number(agent.toolStarts || 0) +
    '/' + Number(agent.completed || 0) + '/' + Number(agent.errors || 0));
  bodyLines.push('sources: ' + Number(agent.sourceCount || 0));
  if (agent.error) bodyLines.push('error: ' + String(agent.error));
  if (summary) bodyLines.push('\nsummary:\n' + summary);
  if (urls.length) {
    bodyLines.push('\nsource urls:');
    for (var i = 0; i < urls.length && i < 10; i++) bodyLines.push('- ' + String(urls[i]));
  }
  bodyEl.textContent = bodyLines.join('\n');
  drawerEl.classList.add('open');
}

export function renderArtifactDrawer(bucket, runId) {
  var drawerEl = document.getElementById('run-artifact-drawer');
  var titleEl = document.getElementById('run-artifact-title');
  var bodyEl = document.getElementById('run-artifact-body');
  if (!drawerEl || !titleEl || !bodyEl) return;
  var sel = getSelectedArtifact();
  if (!bucket || !sel || sel.runId !== runId) {
    drawerEl.classList.remove('open');
    titleEl.textContent = 'artifact';
    bodyEl.textContent = '';
    return;
  }
  var items = Array.isArray(bucket.artifacts) ? bucket.artifacts : [];
  var target = null;
  for (var i = 0; i < items.length; i++) {
    if (String(items[i].id || '') === String(sel.artifactId || '')) {
      target = items[i];
      break;
    }
  }
  if (!target) {
    drawerEl.classList.remove('open');
    titleEl.textContent = 'artifact';
    bodyEl.textContent = '';
    return;
  }
  titleEl.textContent = (target.kind || 'artifact') + ' | ' + (target.source || '-');
  var raw = String(target.rawText || target.text || '').trim();
  if (raw.length > 6000) raw = raw.slice(0, 6000) + '\n...[truncated]...';
  var urlBlock = target.url ? ('\n\nURL:\n' + target.url) : '';
  bodyEl.textContent = raw + urlBlock;
  drawerEl.classList.add('open');
}

export function renderCostCache(bucket) {
  var bodyEl = document.getElementById('run-cost-body');
  if (!bodyEl) return;
  if (!bucket) {
    bodyEl.textContent = 'waiting for llm usage...';
    return;
  }
  var u = bucket.usage || {};
  var provider = String(u.provider || '-');
  var pt = toNumber(u.promptTokens, 0);
  var ct = toNumber(u.completionTokens, 0);
  var tt = toNumber(u.totalTokens, 0);
  var cached = toNumber(u.cachedPromptTokens, 0);
  var ratio = toNumber(u.cacheHitRatio, 0);
  if (ratio <= 0 && pt > 0) ratio = Math.max(0, Math.min(1, cached / pt));
  var budgetLine = '';
  if (typeof u.turnBudget === 'number') {
    budgetLine = '<div><span class="k">turn budget:</span> ' +
      toNumber(u.turnUsed, 0) + '/' + toNumber(u.turnBudget, 0) + '</div>';
  }
  bodyEl.innerHTML =
    '<div><span class="k">provider:</span> ' + escHtml(provider) + '</div>' +
    '<div><span class="k">p/c/t:</span> ' + pt + '/' + ct + '/' + tt + '</div>' +
    '<div><span class="k">cached:</span> ' + cached + ' (' + (ratio * 100).toFixed(1) + '%)</div>' +
    '<div><span class="k">cost:</span> ' + formatUsd(u.estimatedCostUsd) +
    ' | usage events=' + toNumber(u.usageEvents, 0) + '</div>' +
    budgetLine;
}

export function renderArtifacts(bucket) {
  var listEl = document.getElementById('run-artifact-list');
  var emptyEl = document.getElementById('run-artifact-empty');
  if (!listEl || !emptyEl) return;
  if (!bucket) {
    listEl.textContent = '';
    emptyEl.style.display = '';
    emptyEl.textContent = 'no evidence yet';
    updateFilterMeta('artifact-filter-meta', 0, 0, 'artifacts');
    return;
  }
  rebuildArtifactFilterOptions(bucket);
  var items = Array.isArray(bucket.artifacts) ? bucket.artifacts : [];
  var filtered = applyArtifactFilters(items);
  updateFilterMeta('artifact-filter-meta', filtered.length, items.length, 'artifacts');
  listEl.textContent = '';
  var shown = Math.min(filtered.length, 18);
  var sel = getSelectedArtifact();
  var activeArtifactId = (sel && sel.runId === getActiveTimelineRunId())
    ? String(sel.artifactId || '')
    : '';
  for (var i = 0; i < shown; i++) {
    var item = filtered[i] || {};
    var row = document.createElement('div');
    row.className = 'artifact-item' + (activeArtifactId && String(item.id || '') === activeArtifactId ? ' active' : '');
    var urlBlock = '';
    if (item.url) {
      var safeHref = sanitizeHref(item.url);
      var safeDisplay = escHtml(item.url);
      if (safeHref) {
        urlBlock = '<div><a href="' + escHtml(safeHref) + '" target="_blank" rel="noopener noreferrer">' + safeDisplay + '</a></div>';
      } else {
        urlBlock = '<div><span class="m">' + safeDisplay + '</span></div>';
      }
    }
    row.innerHTML =
      '<div><span class="tag">' + escHtml(item.kind || 'artifact') + '</span><span class="src">' +
      escHtml(item.source || '-') + '</span></div>' +
      '<div>' + escHtml(item.text || '') + '</div>' +
      urlBlock;
    (function(runId, artifactId) {
      row.addEventListener('click', function() { selectArtifact(runId, artifactId); });
    })(getActiveTimelineRunId() || '', String(item.id || ''));
    listEl.appendChild(row);
  }
  emptyEl.style.display = shown ? 'none' : '';
  if (!shown) emptyEl.textContent = items.length ? 'no artifacts match current filters' : 'no evidence yet';
}

// ─── Main orchestrator ─────────────────────────────────────────────
export function renderRunTimeline(runId) {
  var store = getRunEventStore();
  var bucket = store[runId];
  var titleEl = document.getElementById('run-panel-title');
  var summaryEl = document.getElementById('run-summary');
  var listEl = document.getElementById('run-timeline-list');
  var emptyEl = document.getElementById('run-timeline-empty');
  if (!titleEl || !summaryEl || !listEl || !emptyEl) return;

  if (!bucket) {
    titleEl.textContent = 'run timeline';
    summaryEl.textContent = 'no run events yet';
    renderRunHistory();
    renderRoutingTrace(null);
    renderAgentBoard(null);
    renderCostCache(null);
    renderArtifacts(null);
    renderArtifactDrawer(null, '');
    rebuildTimelineFilterOptions(null);
    updateFilterMeta('timeline-filter-meta', 0, 0, 'events');
    listEl.textContent = '';
    emptyEl.style.display = '';
    emptyEl.textContent = 'waiting for event envelope...';
    return;
  }

  var keys = Object.keys(bucket.counters || {});
  var doneCount = Number(bucket.counters.done || 0);
  var errCount = Number(bucket.counters.error || 0);
  var toolCount = Number(bucket.counters.tool_start || 0);
  var subCount = Number(bucket.counters.sub_complete || 0);
  var totalEvents = (bucket.events || []).length;
  var uptime = Math.max(0, Math.round((Date.now() - bucket.startedAt) / 1000));
  var queryText = bucket.query ? ('<div><span class="k">query:</span> ' + escHtml(bucket.query) + '</div>') : '';

  titleEl.textContent = runId === 'adhoc' ? 'run timeline' : ('run #' + runId.slice(0, 6));
  summaryEl.innerHTML =
    '<div><span class="k">state:</span> ' + escHtml(bucket.lastState || 'idle') + '</div>' +
    '<div><span class="k">events:</span> ' + totalEvents + ' | types ' + keys.length + '</div>' +
    '<div><span class="k">tools/sub:</span> ' + toolCount + '/' + subCount +
    ' | <span class="k">done/error:</span> ' + doneCount + '/' + errCount + '</div>' +
    '<div><span class="k">age:</span> ' + uptime + 's</div>' +
    queryText;

  renderRunHistory();
  renderRoutingTrace(bucket);
  renderAgentBoard(bucket);
  renderCostCache(bucket);
  renderArtifacts(bucket);
  renderArtifactDrawer(bucket, runId);
  rebuildTimelineFilterOptions(bucket);

  listEl.textContent = '';
  var events = bucket.events || [];
  var filteredEvents = applyTimelineFilters(events);
  updateFilterMeta('timeline-filter-meta', filteredEvents.length, events.length, 'events');
  var start = Math.max(0, filteredEvents.length - 40);
  for (var i = filteredEvents.length - 1; i >= start; i--) {
    var item = filteredEvents[i];
    var el = document.createElement('div');
    el.className = 'timeline-item';
    el.setAttribute('tabindex', '0');
    el.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        this.click();
      }
    });
    var seqText = (item.seq === null || typeof item.seq === 'undefined') ? '-' : String(item.seq);
    var sourceText = item.source && item.source !== '-' ? item.source : item.agent;
    var detailText = item.detail ? ('<div>' + escHtml(item.detail) + '</div>') : '';
    el.innerHTML =
      '<div class="h"><span>' + escHtml(item.eventType) + '</span><span class="seq">#' + escHtml(seqText) + '</span></div>' +
      '<div class="src">' + escHtml(sourceText || '-') + '</div>' +
      detailText;
    listEl.appendChild(el);
  }
  emptyEl.style.display = filteredEvents.length ? 'none' : '';
  if (!filteredEvents.length) {
    emptyEl.textContent = events.length ? 'no timeline events match current filters' : 'waiting for event envelope...';
  }
}

// ─── Combined clear (store + DOM) ─────────────────────────────────
export function clearRunTimelineAndRender() {
  clearRunStore();
  var artInput = document.getElementById('artifact-text-filter');
  if (artInput) artInput.value = '';
  setTimelineLiveMode(true);
  renderRunTimeline('');
}
