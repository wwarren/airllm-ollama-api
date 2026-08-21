# airllm-ollama-api

An Ollama-compatible HTTP API in front of [AirLLM](https://github.com/lyogavin/airllm),
so anything that already speaks Ollama — Open WebUI, LiteLLM, editor plugins,
plain `curl` — can drive a layer-swapped model that does not fit in VRAM.

AirLLM streams a model's layers through memory one at a time, which is what
lets a 70B run on a machine that could never hold it at once. The tradeoff is
speed: expect seconds to minutes per token, not tokens per second. This wrapper
is built around that reality — one generation at a time, generous timeouts, and
a background model load so clients can connect while weights are still arriving.

## Layout

| File | Purpose |
| --- | --- |
| `airllm-ollama-api.py` | The API. Runs standalone (`python airllm-ollama-api.py`) or under systemd. |
| `airllm-olllama-api-install.sh` | Installs to `/opt/airllm-ollama-api` in a venv and registers the service. |
| `requirements.txt` | Pinned floor versions for the inference and HTTP stack. |
| `env.example` | Every setting, with defaults and notes. |
| `tests/` | Fast tests against a stubbed model — no weights downloaded. |

## Install

```bash
git clone <this repo> && cd airllm-openai-api-wrapper
cp env.example .env       # edit before serving anything real
./airllm-olllama-api-install.sh
```

That registers **`airllm-ollama-api.service`**, installed under
`/opt/airllm-ollama-api`. `systemctl status airllm-ollama-api` describes it as
"Ollama-compatible HTTP API served by AirLLM" with the port it's listening on,
so it's unambiguous next to a real `ollama.service`. Override either name with
`SERVICE_NAME=... APP_DIR=... ./airllm-olllama-api-install.sh`.

```bash
systemctl status airllm-ollama-api      # what it is, and whether it's up
journalctl -u airllm-ollama-api -f      # model load progress and requests
sudo systemctl restart airllm-ollama-api
```

`airllm-olllama-api-install.sh` refuses to run as root, checks that systemd is actually running,
warns if `ollama.service` already owns port 11434, builds a virtualenv, verifies
every module imports, and only then writes and starts the unit. Re-running is
safe — it refreshes code and dependencies and leaves your `.env` alone.

To run it without systemd:

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
set -a; source .env; set +a
./venv/bin/python airllm-ollama-api.py
```

## Configure

Everything is environment variables; `env.example` is the full list. The four
that matter most:

- **`AIRLLM_MODEL_ID`** — defaults to `Qwen/Qwen2.5-3B-Instruct`. Start there.
  Validate that your client connects and streams, *then* switch to
  `meta-llama/Meta-Llama-3-70B-Instruct`. Debugging a wrapper at one token per
  minute is miserable.
- **`AIRLLM_DEVICE`** — defaults to `cpu`. AirLLM's own default is `cuda:0`, so
  this has to be set explicitly for RAM-based swapping.
- **`AIRLLM_MAX_SEQ_LEN`** — defaults to 2048. AirLLM's internal default is
  **512**, which silently truncates longer prompts.
- **`HF_TOKEN`** — required for gated repos. Llama 3 is gated; without a token
  the load fails with a 401 well into startup.

`HF_HOME` is worth setting too. A 70B is roughly 150 GB of weights plus the
sharded copy AirLLM writes, and by default all of it lands in `~/.cache`.

## Endpoints

| Endpoint | Notes |
| --- | --- |
| `GET /` | Returns `Ollama is running` — several clients probe this first. |
| `GET /api/version` | Reports `AIRLLM_OLLAMA_VERSION`; Open WebUI feature-gates on it. |
| `GET /api/tags` | One model, the configured one. `format` is `safetensors`, not `gguf`. |
| `POST /api/show` | Model metadata and capabilities. |
| `GET /api/ps` | Empty until loaded, then one entry with `size_vram: 0`. |
| `POST /api/chat` | Streaming NDJSON or `"stream": false`. |
| `POST /api/generate` | Same, with `raw` to skip chat templating. |
| `POST /api/embeddings`, `/api/embed` | `501` — AirLLM's layer swapping doesn't serve embeddings. |
| `GET /health` | Non-Ollama. Load status, device, and the load error if there was one. |

Response packets carry `created_at`, `done_reason`, and the full set of
`*_count` / `*_duration` fields, because LiteLLM builds its usage numbers from
`prompt_eval_count` and `eval_count`, and expects `message` on *every* chat
packet including the final one.

### Client setup

**Open WebUI** — add `http://<host>:11434` as an Ollama connection. It calls
`/api/version` and `/api/tags` on connect, both of which answer immediately even
while weights load. Raise the request timeout; its default will give up long
before a 70B produces its first token.

**LiteLLM** — use the `ollama_chat/` prefix, which maps onto `/api/chat`:

```python
completion(model="ollama_chat/Qwen/Qwen2.5-3B-Instruct",
           api_base="http://localhost:11434",
           messages=[{"role": "user", "content": "hi"}])
```

Set `AIRLLM_MODEL_ALIAS` if you want a shorter name in the model picker.

## What it deliberately doesn't do

- **No authentication.** `AIRLLM_HOST` defaults to `127.0.0.1` for that reason.
  Only bind `0.0.0.0` behind a firewall or an authenticating proxy.
- **No concurrency.** AirLLM swaps layers through one shared set of buffers, so
  requests are serialized behind a lock. A second request waits; it does not
  corrupt the first.
- **No model switching.** One process serves one model. `/api/pull`,
  `/api/create`, `/api/delete` are not implemented — there's nothing to pull.
- **No tool calling or embeddings.**
- **No OpenAI `/v1/*` routes.** Point an OpenAI-shaped client at LiteLLM or
  Open WebUI in front of this, and let it do the translation.

## Tests

```bash
pip install pytest httpx fastapi
pytest -q
```

`tests/fakes.py` injects stand-in `airllm` and `transformers` modules before
`airllm-ollama-api.py` is imported, so the suite runs in about a second and downloads
nothing. That works only because `airllm-ollama-api.py` imports both lazily — if someone
hoists those imports to module scope, the tests fail immediately, on purpose.

The suite covers packet shapes for both endpoints in both modes, the chat
template, option mapping, error propagation, 503-while-loading, that
generation always passes `use_cache=True`, that only one generation runs at a
time, and that abandoning a stream actually cancels the model.

## Operating notes

- **First run is very slow and looks hung.** AirLLM downloads the model and
  writes per-layer shards before it can generate anything. `journalctl -u
  airllm-ollama-api -f` shows progress; `curl localhost:11434/health` reports
  `loading`, `ready`, or `error`.
- Requests return **503** while the model is still loading, not a hang. Tune
  `AIRLLM_LOAD_WAIT`.
- **`AIRLLM_COMPRESSION=4bit`** needs `bitsandbytes`, which is CUDA-first.
  Verify it works on a CPU-only host before depending on it; unset is the
  safe default.
- The unit uses `Restart=on-failure` with `StartLimitBurst=5`, so a
  misconfiguration (bad model ID, missing token) stops after five attempts
  instead of re-downloading weights forever.
- **WSL**: systemd is off unless you set `systemd=true` under `[boot]` in
  `/etc/wsl.conf` and run `wsl --shutdown`. `airllm-olllama-api-install.sh` checks and tells you.
