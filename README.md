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
| `airllm_ollama_api.py` | The API. Runs standalone (`python airllm_ollama_api.py`) or under systemd. |
| `proxmox-lxc-install.sh` | Proxmox VE host-side LXC creator that runs the system installer inside the container. |
| `airllm-system-install.sh` | One-shot Linux host bootstrap: OS packages, service user, caches, then API install. |
| `airllm-ollama-api-install.sh` | Installs to `/opt/airllm-ollama-api` in a venv and registers the service. |
| `requirements.txt` | Pinned floor versions for the inference and HTTP stack. |
| `env.example` | Every setting, with defaults and notes. Copy to `.env`. |
| `.gitignore` | Keeps `.env` (which holds `HF_TOKEN`), caches and shards out of git. |
| `tests/` | Fast tests against a stubbed model — no weights downloaded. |

## Install

```bash
git clone <this repo> && cd airllm-ollama-api
cp env.example .env       # edit before serving anything real
./airllm-ollama-api-install.sh
```

On a fresh Linux host, run the system bootstrapper instead:

```bash
./airllm-system-install.sh
```

It installs Python/build tools with the detected package manager, creates a
dedicated `airllm` service account, prepares `HF_HOME` and
`AIRLLM_LAYER_SHARDS_PATH` under `/opt/airllm-ollama-api`, then runs
`airllm-ollama-api-install.sh`. Override paths and names the same way:
`APP_DIR=... SERVICE_USER=... ./airllm-system-install.sh`.

If `nvidia-smi` sees an NVIDIA GPU, the bootstrapper sets a new `.env` to
`AIRLLM_DEVICE=cuda:0`. The API installer then installs a CUDA-enabled `torch`
wheel into `/opt/airllm-ollama-api/venv`, verifies
`torch.cuda.is_available()`, and installs `bitsandbytes` for 4-bit/8-bit
compression before starting the service. Disable those with
`INSTALL_CUDA_TORCH=0` or `INSTALL_BITSANDBYTES=0`; change the PyTorch wheel
index with `TORCH_CUDA_INDEX_URL=...` when you need a different CUDA build.

On a Proxmox VE node, create an LXC container and install the service into it:

```bash
./proxmox-lxc-install.sh
```

It prompts for the LXC template, VMID, hostname, CPU cores, RAM, swap,
container-enabled storage target, root disk size, and network settings. Static
networking is the default: it asks for IP address, subnet mask or CIDR prefix,
and default gateway. DHCP is available as a menu option. It then creates and
starts the container, copies this repo into the guest, and runs
`airllm-system-install.sh` inside the container. Defaults can be overridden
with environment variables such as `DEFAULT_CORES=8`, `DEFAULT_MEMORY_MB=65536`,
`DEFAULT_DISK_GB=500`, `DEFAULT_IP_ADDRESS=192.168.1.50`,
`DEFAULT_SUBNET=24`, `DEFAULT_GATEWAY=192.168.1.1`, and `APP_DIR=...`.
Use an Ubuntu LTS or Debian template for the smoothest install; development
templates such as Ubuntu `resolute` may not have stable apt Release files yet.
The installer waits for a default route and DNS in the container before it
starts package installation.

That registers **`airllm-ollama-api.service`**, installed under
`/opt/airllm-ollama-api`. `systemctl status airllm-ollama-api` describes it as
"Ollama-compatible HTTP API served by AirLLM" with the port it's listening on,
so it's unambiguous next to a real `ollama.service`. Override either name with
`SERVICE_NAME=... APP_DIR=... ./airllm-ollama-api-install.sh`.

```bash
systemctl status airllm-ollama-api      # what it is, and whether it's up
journalctl -u airllm-ollama-api -f      # model load progress and requests
sudo systemctl restart airllm-ollama-api
```

It runs either as a normal user with `sudo`, or directly as root — which is
what you want inside a container. When it detects root it skips `sudo`
entirely; if the service would then also run as root it says so, since the API
has no authentication. Give it a dedicated account instead with
`SERVICE_USER=airllm ./airllm-ollama-api-install.sh`.

Before touching anything it checks that systemd is actually running, warns if
`ollama.service` already owns port 11434, builds a virtualenv, and verifies
every module imports. Re-running is safe — it refreshes code and dependencies
and leaves your `.env` alone.

The last step verifies the install rather than assuming it worked: it confirms
the unit is active, polls `/api/version` until the HTTP server answers (up to
60s), posts `/api/pull` to initiate the configured model download/sharding,
prints `/health` so you can see whether the weights are still loading, and
shows `systemctl status`. If any of that fails it dumps the last 40 journal
lines and exits non-zero.

```
[7/7] Verifying the service
  unit is active
  waiting for the API on 127.0.0.1:11434 
  /api/version -> {"version":"0.32.15"}
  initiating model download/sharding for Qwen/Qwen2.5-72B-Instruct
  /api/pull    -> {"status":"success"}
  /health      -> {"status":"loading", ...}
```

To run it without systemd:

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
set -a; source .env; set +a
./venv/bin/python airllm_ollama_api.py
```

## Configure

Everything is environment variables; `env.example` is the full list. The four
that matter most:

- **`AIRLLM_MODEL_ID`** — defaults to `Qwen/Qwen2.5-72B-Instruct`. Set this to
  a smaller Qwen model before installing if you want a quick smoke test instead
  of beginning the large first download immediately.
- **`AIRLLM_DEVICE`** — defaults to `cpu`, except `airllm-system-install.sh`
  uses `cuda:0` for a new `.env` when `nvidia-smi` detects an NVIDIA GPU.
  AirLLM's own default is `cuda:0`, so this has to be set explicitly for
  RAM-based swapping.
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
| `POST /api/pull` | Accepts the configured model name or alias and starts the background load. |
| `POST /api/chat` | Streaming NDJSON or `"stream": false`. |
| `POST /api/generate` | Same, with `raw` to skip chat templating. |
| `POST /api/embeddings`, `/api/embed` | `501` — AirLLM's layer swapping doesn't serve embeddings. |
| `POST /api/create`, `/api/copy`, `/api/push`, `/api/delete` | `501` — one configured AirLLM model per process. |
| `GET /health` | Non-Ollama. Load status, device, and the load error if there was one. |

Response packets carry `created_at`, `done_reason`, and the full set of
`*_count` / `*_duration` fields, because LiteLLM builds its usage numbers from
`prompt_eval_count` and `eval_count`, and expects `message` on *every* chat
packet including the final one.

Requests naming any model other than `AIRLLM_MODEL_ID`, `AIRLLM_MODEL_ALIAS`,
or their implicit `:latest` forms return `404`. Omitting `model` keeps the
single-model convenience path for lightweight clients and tests.

### Client setup

**Open WebUI** — add `http://<host>:11434` as an Ollama connection. It calls
`/api/version` and `/api/tags` on connect, both of which answer immediately even
while weights load. Raise the request timeout; its default will give up long
before a 70B produces its first token.

**LiteLLM** — use the `ollama_chat/` prefix, which maps onto `/api/chat`:

```python
completion(model="ollama_chat/Qwen/Qwen2.5-72B-Instruct",
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
  `/api/create`, `/api/delete` are compatibility shims; they do not change the
  configured model.
- **No tool calling or embeddings.**
- **No OpenAI `/v1/*` routes.** Point an OpenAI-shaped client at LiteLLM or
  Open WebUI in front of this, and let it do the translation.

## Tests

```bash
pip install pytest httpx fastapi
pytest -q
```

`tests/fakes.py` injects stand-in `airllm` and `transformers` modules before
`airllm_ollama_api.py` is imported, so the suite runs in about a second and downloads
nothing. That works only because `airllm_ollama_api.py` imports both lazily — if someone
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
- **`AIRLLM_COMPRESSION=4bit`** needs `bitsandbytes`. The system bootstrapper
  installs it automatically when an NVIDIA GPU is detected; force it with
  `INSTALL_BITSANDBYTES=1` or skip it with `INSTALL_BITSANDBYTES=0`. Unset
  compression is the safe default.
- The unit uses `Restart=on-failure` with `StartLimitBurst=5`, so a
  misconfiguration (bad model ID, missing token) stops after five attempts
  instead of re-downloading weights forever.
- **`.env` is gitignored and `env.example` is not.** The token lives in
  `.env`; the template is the tracked file. Don't rename either.
- **Containers**: run the installer as root. It needs systemd as PID 1 —
  a plain `docker run` image without an init won't work.
- **WSL**: systemd is off unless you set `systemd=true` under `[boot]` in
  `/etc/wsl.conf` and run `wsl --shutdown`. `airllm-ollama-api-install.sh` checks and tells you.
