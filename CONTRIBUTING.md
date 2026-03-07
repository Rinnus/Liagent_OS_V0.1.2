# Contributing

Thanks for contributing to LiAgent.

The project is moving quickly, so favor changes that are small, verifiable, and easy to roll back.

## 1. Development environment

```bash
python3 -m venv venv
./venv/bin/pip install -e .
```

Optional extras:

```bash
./venv/bin/pip install -e '.[browser,mcp,discord]'
```

## 2. Local validation

Before opening a PR, run at least the fast regression suite aligned with CI:

```bash
PYTHONPATH=src ./venv/bin/python -m compileall -q src
PYTHONPATH=src ./venv/bin/python -m pytest -q \
  tests/test_events.py \
  tests/test_skill_router_simplified.py \
  tests/test_health.py \
  tests/test_brain_layers.py \
  tests/test_web_event_envelope.py \
  tests/test_tool_parsing.py
```

If your change touches policy, orchestration, task execution, tool execution, or integration flows, run the full test suite:

```bash
PYTHONPATH=src ./venv/bin/python -m pytest -q
```

Current test layout:

- `tests/`: core unit and regression coverage
- `tests/integration/`: browser, Discord, voice, VLM, and end-to-end integration coverage
- `tests/manual/`: live/manual validation scripts that are not part of the default pytest run

If your change targets browser, Discord, voice, or other environment-heavy paths, run the integration slice that matches your area. Some integration tests require optional extras such as `.[browser]` or `.[discord]`.

```bash
PYTHONPATH=src ./venv/bin/python -m pytest -q tests/integration
```

## 3. PR expectations

- Keep each PR focused on a single theme whenever possible.
- Add tests for behavior changes.
- Update documentation for user-visible changes.
- Never commit real keys, personal paths, databases, virtual environments, or other local artifacts.

## 4. Issues and onboarding feedback

- Use the GitHub issue forms for bug reports, feature requests, and documentation or onboarding feedback.
- Bug reports should include the runtime mode, the interface used (`CLI`, `Web`, or `Discord`), and the smallest reliable reproduction.
- If the problem is "I could not get the first run working," link the exact step from `docs/getting-started.md` that failed.
- If the report is about current product boundaries rather than a defect, check `docs/current-limitations.md` first and then open a feature request if the gap is intentional but important.

## 5. Commit message style

Prefer short Conventional Commit messages, for example:

- `feat(agent): add heartbeat confirmation callback`
- `fix(security): enforce auth header on ws handshake`
- `docs(readme): refresh runtime mode and routing section`

## 6. Code and design principles

- Safety policy beats "it seems to work."
- Preserve backward compatibility by default; document any migration path when you cannot.
- High-risk tool paths such as writes, execution, and network operations must remain auditable.
