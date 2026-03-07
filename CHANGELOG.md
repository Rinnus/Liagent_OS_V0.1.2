# LiAgent OS Changelog

This file records notable project changes.

## 0.1.2 - 2026-03-07

### Added

- Heartbeat confirmation loop:
  - `POST /api/tasks/confirm`
  - approve / reject actions in Web and Discord
  - expiration sweeps for pending confirmation items
- Public collaboration and onboarding assets:
  - GitHub issue forms for bugs, feature requests, and docs or onboarding feedback
  - a pull request template for release-aware reviews
  - `SUPPORT.md`, `docs/getting-started.md`, and `docs/current-limitations.md`
- Stronger execution recovery:
  - failure taxonomy
  - global retry budget tracking
  - checkpoint matching and resume
  - evidence pinning before resuming execution
- Better multi-agent observability:
  - progressive synthesis events
  - routing decision metadata
  - fuller runtime traces
- Stronger safety and governance:
  - tighter auth-header and budget scoping
  - fail-closed heartbeat budget gates
  - grounding gate integration in the execution path

### Changed

- The main execution path now wires `orchestrator`, `harness`, and `knowledge` together, reducing "present but not connected" modules.
- Task queue and budget propagation now cover the full enqueue-to-run flow.
- Candidate actions are now structured as `tool_name` / `tool_args`, with stricter parse validation and deduplication.
- Project documentation was refreshed to reflect the current runtime shape and public-repo positioning.
- Public-facing naming and copy now consistently use **LiAgent OS**.
- The public documentation set is now centered on a public architecture overview and roadmap rather than internal planning and release process notes.
- The public documentation set now also includes a first-run guide and a clear statement of current public boundaries.

### Fixed

- Multiple P0/P1/P2 heartbeat integration issues, including duplicate execution, budget edge cases, and pending-confirmation handling.
- Tool execution and timeout edge cases that could truncate internal timeouts incorrectly.
- Several fail-open paths in execution flows, tightened into more conservative failure behavior.
