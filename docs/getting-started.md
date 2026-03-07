# LiAgent OS Getting Started

This is the shortest path to a first successful run of LiAgent OS.

Recommended starting point: **API bootstrap mode**. It lets you confirm that the runtime works before you invest time in local models, model paths, TTS/STT assets, or hybrid tuning.

## What you need

- Python `3.10+`
- one compatible LLM API key
- about 5 minutes

## Success target

By the end of this guide, you should be able to:

- start the Web UI on `http://127.0.0.1:8080`
- send a prompt and receive a response
- see runtime events in the interface

## 1. Install the project

```bash
cd /path/to/liagent_git
python3 -m venv venv
./venv/bin/pip install -e .
```

Optional extras can wait until later:

```bash
./venv/bin/pip install -e '.[browser]'
./venv/bin/pip install -e '.[discord]'
./venv/bin/pip install -e '.[mcp]'
```

## 2. Create local config files

```bash
cp config.example.json config.json
cp .env.example .env
```

## 3. Add only the minimum API settings

Put your API key in `.env`:

```bash
LLM_API_KEY=your_api_key
```

Optional overrides:

```bash
# LLM_API_BASE_URL=https://api.openai.com/v1
# LLM_API_MODEL=gpt-4o
```

You do not need to configure local model paths for this first run.

## 4. Start the Web UI

```bash
./venv/bin/liagent --web --host 127.0.0.1 --port 8080
```

Then open [http://127.0.0.1:8080](http://127.0.0.1:8080).

## 5. Confirm the first successful run

You should now see:

- the Web UI loading on localhost
- a response when you send a simple message
- runtime events, not just a plain chat bubble

Try one of these first prompts:

- `Summarize the AI agent releases from this week that are worth tracking.`
- `Break this request into an execution plan, research first, then give me the final proposal.`
- `Ask me before any file writes, command execution, or higher-risk actions.`

## 6. If it does not work

Check these first:

- Missing response: verify `LLM_API_KEY` is set in `.env`
- Startup path validation errors: keep the first run in API bootstrap mode and do not add local model paths yet
- Browser/tooling features missing: install the `.[browser]` extra and finish the Playwright setup later
- Discord confirmations missing: add the `.[discord]` extra and verify the required Discord environment variables

## 7. What to do next

After the first successful run:

- move to [README(EN).md](../README%28EN%29.md) for the full capability map
- read [docs/current-limitations.md](current-limitations.md) for the current public boundaries
- read [docs/architecture.md](architecture.md) for the runtime structure
- switch to `local_private` or `hybrid_balanced` only after the basic path is working
