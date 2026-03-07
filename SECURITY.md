# Security Policy

## Supported scope

Security fixes are guaranteed only for the latest code on the `main` branch.

## Reporting vulnerabilities

Do not disclose sensitive vulnerabilities in public issues.

Preferred reporting channels:

1. GitHub Private Security Advisory
2. The maintainer contact listed on the repository profile

Please include:

- impact and severity
- the smallest reliable reproduction
- affected modules or files
- optional mitigation ideas

## Secrets and sensitive data

- Never commit real API keys, tokens, or credentials.
- Keep secrets in local `.env` files and use `.env.example` only as a placeholder template.
- Treat `config.json` as local runtime state, not as a place to commit real secrets.
- If you suspect a leak, immediately:
  1. rotate the credential
  2. clean or rewrite history if necessary
  3. publish mitigation guidance and follow-up steps

## Submission baseline

- Run the minimum CI-aligned regression suite.
- Verify that high-risk tool paths still require confirmation and auditing.
- Scan the codebase and recent history for common secret patterns.

## Response priorities

- Fix remotely triggerable issues, privilege escalation, and data exposure first.
- Add regression coverage after the fix so the same class of issue does not return.
