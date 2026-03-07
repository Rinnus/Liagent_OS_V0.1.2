/**
 * Status bar renderer — run status dot + label.
 * C1: Reads stores via getters, writes to DOM only.
 * Never mutates another domain's store (updates activeTimelineRunId
 * via setter only when inline code previously did direct assignment).
 */
import {
  getActiveServiceTier,
  getActiveSkillName,
} from '../stores/chat-store.js';

import {
  getRunEventStore,
  getActiveTimelineRunId,
  getPinnedTimelineRunId,
  setActiveTimelineRunId,
} from '../stores/run-store.js';

import { renderRunTimeline, renderRunHistory } from './run-panel.js';

// ─── Status bar ─────────────────────────────────────────────────────

export function setRunStatus(state, runId) {
  var status = document.getElementById('status-run');
  var dot = document.getElementById('dot-run');
  if (!status || !dot) return;
  var label = state || 'idle';
  var tierText = getActiveServiceTier() ? (' | ' + getActiveServiceTier()) : '';
  var skillText = getActiveSkillName() ? (' | ' + getActiveSkillName()) : '';
  status.textContent = runId
    ? (label + ' #' + runId.slice(0, 6) + tierText + skillText)
    : (label + tierText + skillText);
  dot.className = (
    label === 'streaming' || label === 'queued' || label === 'accepted'
  ) ? 'dot on' : 'dot';
  if (runId) {
    if (!getPinnedTimelineRunId()) setActiveTimelineRunId(runId);
    var store = getRunEventStore();
    if (store[runId] && (!getPinnedTimelineRunId() || getPinnedTimelineRunId() === runId)) {
      renderRunTimeline(runId);
    } else {
      renderRunHistory();
    }
  }
}
