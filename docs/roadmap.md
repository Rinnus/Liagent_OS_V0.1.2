# LiAgent OS Roadmap

This roadmap is intended for the public repository. It focuses on durable direction rather than internal implementation checklists.

LiAgent OS is not just pursuing a better agent runtime in the abstract. The long-term destination is a trusted, all-around personal AI agent that can help users think, monitor, organize, and act across everyday workflows while remaining local-first, privacy-conscious, and safety-governed.

## Current release posture

The recommended public baseline is **`v0.1.2 (alpha)`**.

That label is reasonable because:

- the runtime already supports real interaction, tools, task execution, and governance loops
- the project is still pre-1.0 and makes no claim of API or workflow stability
- this release is an incremental public-facing hardening step, not a platform reset

When a move to `0.2.0` would make more sense:

- the public contract changes materially
- the default operating mode changes
- external integrators need to treat the release as a new baseline

## Roadmap themes

### 1. Toward an all-around personal AI agent

Priority:

- stronger continuity across sessions, tasks, goals, and monitoring
- better follow-through across research, organization, and action
- clearer product experience as a long-lived personal assistant rather than a single-turn assistant

### 2. Stronger local-model experience

Priority:

- better local bootstrap paths
- smoother model validation and startup guidance
- more dependable local-first defaults on supported hardware
- broader real-world use of local models where user trust and data sensitivity matter most

### 3. Privacy and security as product pillars

Priority:

- stronger privacy-first defaults
- tighter safety controls around high-risk actions
- clearer trust, approval, and audit boundaries as core product behavior

### 4. Better hybrid routing

Priority:

- clearer routing across local and cloud backends
- better latency and cost tradeoffs
- more stable long-context handling and delegation paths

### 5. Long-running execution reliability

Priority:

- checkpoint and resume hardening
- better retry behavior and failure classification
- more consistent execution outcomes across multi-step runs

### 6. Governed semi-autonomy

Priority:

- stronger heartbeat behavior
- cleaner suggestion and monitoring flows
- safer goal creation and reflection loops

### 7. Public open-source usability

Priority:

- clearer architecture and onboarding docs
- lower-friction first-run path
- better contributor-facing documentation

## What the project is not promising yet

LiAgent OS is not positioning itself as:

- a zero-config consumer SaaS product
- a stable enterprise platform with frozen interfaces
- a fully autonomous system that should act without governance

The public promise is narrower and more credible: a local-first agent runtime that already works, exposes its reasoning and actions more clearly than typical chat wrappers, and is being hardened for long-lived real workflows.
