"""AnomalyPreCheck — heartbeat pre-check that surfaces recent anomaly events.

Bridges the attention system (AnomalyDetector) with the heartbeat system
(HeartbeatRunner) so that detected anomalies influence heartbeat decisions.
"""

import time

from .heartbeat import PreCheckResult


class AnomalyPreCheck:
    """Heartbeat pre-check that buffers and drains anomaly events.

    Wire ``on_anomaly`` as a callback from AnomalyDetector. The heartbeat
    runner calls ``check()`` periodically, draining any pending anomalies.
    """

    source_type = "anomaly_detector"

    def __init__(self):
        self._pending: list[dict] = []

    async def check(self, cursor: str | None = None) -> PreCheckResult:
        """Return pending anomalies as a PreCheckResult."""
        if not self._pending:
            return PreCheckResult(
                has_changes=False,
                change_summary="",
                new_cursor=cursor or "",
            )
        batch = self._pending.copy()
        self._pending.clear()
        summaries = [a.get("summary", "anomaly detected") for a in batch]
        return PreCheckResult(
            has_changes=True,
            change_summary="; ".join(summaries),
            new_cursor=str(time.time()),
        )

    def on_anomaly(self, anomaly: dict):
        """Callback — wired to AnomalyDetector's on_anomaly chain."""
        self._pending.append(anomaly)
