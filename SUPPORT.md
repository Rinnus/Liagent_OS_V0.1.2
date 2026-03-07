# Support

LiAgent OS is an alpha open-source project. Support is best-effort, and the fastest path to a useful answer is a narrow, reproducible report with the right context.

## Before opening an issue

Start with these public docs:

- [README(EN).md](README%28EN%29.md) for the product overview and recommended paths
- [docs/getting-started.md](docs/getting-started.md) for the first successful run
- [docs/current-limitations.md](docs/current-limitations.md) for the current public boundaries
- [SECURITY.md](SECURITY.md) for vulnerability reporting

## Which path to use

- Reproducible defect: open the `Bug report` issue form
- Product gap or roadmap ask: open the `Feature request` issue form
- Confusing setup or unclear docs: open the `Docs / onboarding feedback` issue form
- Security issue: do not open a public issue; follow [SECURITY.md](SECURITY.md)

## What to include

Useful reports usually include:

- LiAgent OS version
- operating system and Python version
- runtime mode such as `local_private`, `hybrid_balanced`, or API bootstrap
- interface used: `CLI`, `Web`, or `Discord`
- the smallest reliable reproduction
- logs, screenshots, or traces with secrets removed

## Current support stance

- The recommended first run is API bootstrap mode.
- Local-first setups are important to the project direction, but they still require more manual setup than the API bootstrap path.
- Public interfaces and workflow details may still change during the alpha phase.
- High-risk behavior is expected to remain governed by approvals, policies, and auditability.
