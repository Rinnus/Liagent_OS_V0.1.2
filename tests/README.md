# Test Layout

- `tests/`: core unit and regression coverage used by the default pytest run
- `tests/integration/`: browser, Discord, VLM, STT, and broader end-to-end coverage
- `tests/manual/`: live or operator-driven validation scripts outside default pytest collection

The public repository keeps most tests because they increase trust in the runtime. The split is meant to make the structure easier to scan, not to hide important coverage.
