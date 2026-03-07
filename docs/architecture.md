# LiAgent OS Architecture

LiAgent OS is a local-first runtime for governed AI agents. The architecture is designed around long-lived execution, controlled autonomy, and clear operational boundaries rather than one-shot chat alone.

## System shape

At a high level, LiAgent OS combines:

- interaction surfaces: CLI, Web, and Discord
- model backends: local inference, cloud APIs, and hybrid routing
- execution loops: conversation, planning, task queues, heartbeat, and reflection
- governance layers: policy gates, confirmation flows, trust registry, budgets, and audit logs

## Core runtime loops

### 1. Conversation loop

The main agent loop handles:

- user input parsing
- prompt construction with long-term context
- tool selection and execution
- plan tracking for multi-step work
- result synthesis and event streaming

This is the loop users see most directly in CLI, Web, and Discord.

### 2. Task and scheduling loop

LiAgent OS can continue work over time through:

- delayed one-shot tasks
- cron-style recurring tasks
- file-triggered tasks
- resumable queued runs

This turns the runtime into something closer to an operating layer for agents than a single request-response shell.

### 3. Heartbeat loop

The heartbeat loop performs periodic checks, recalls relevant evidence, proposes bounded actions, and enforces allowlists and risk gates before anything executes.

It is the main bridge from passive assistant behavior to governed semi-autonomy.

### 4. Goal and reflection loop

Higher-level pattern detection and reflection can group repeated behavior, identify opportunities for follow-through, and propose new goals or suggestions.

This layer is still alpha, but it already defines the direction of the project: agents that can keep context across time without losing control boundaries.

## Routing model

LiAgent OS is built around three practical runtime modes:

- `local_private`: privacy-first, local execution first
- `hybrid_balanced`: pragmatic default for mixed workloads
- `cloud_performance`: capability-first for heavier tasks

The long-term direction is not "local or cloud" as a binary choice. It is reliable routing across local models and hybrid local-cloud model services, with explicit cost, latency, and governance tradeoffs.

## Safety and governance

The system treats safety as part of the runtime, not as post-processing.

Key controls include:

- tool profiles such as `minimal`, `research`, and `full`
- trust registry for MCP and other expandable tool surfaces
- confirmation flows for high-risk actions
- working-directory boundaries and sandbox-backed execution
- token and cost budgets
- audit records and runtime event traces

## Public module map

```text
src/liagent/
  agent/          core loops, long-running execution, recovery, reflection
  engine/         model routing for LLM, TTS, and STT backends
  knowledge/      retrieval, evidence, and knowledge injection
  orchestrator/   multi-agent coordination and synthesis
  tools/          tool registry, policy, executors, trust controls
  ui/             CLI, Web, Discord, and API surfaces
  voice/          voice-related capabilities
  skills/         skill routing and assets
  utils/          shared support utilities
```

## Current architectural stance

LiAgent OS is already a real runtime, but it is still in alpha. The architecture is strongest where it connects execution, governance, and observability. The next step is to make that foundation more reliable for local models and hybrid model services under practical long-running workloads.
