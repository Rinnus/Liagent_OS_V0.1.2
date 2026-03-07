#!/usr/bin/env python3
"""Print weekly self-supervision metrics from SQLite."""

from __future__ import annotations

import argparse
import json

from liagent.agent.self_supervision import InteractionMetrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    metrics = InteractionMetrics()
    report = metrics.weekly_summary(days=max(1, min(args.days, 30)))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
