"""Ollama-compatible HTTP API backed by AirLLM layer-swapped inference.

Speaks enough of the Ollama wire protocol that Open WebUI, LiteLLM, and
anything else expecting an Ollama endpoint can point at it unchanged.

Configuration is entirely through environment variables; see env.example.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.getenv("AIRLLM_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("airllm-ollama-api")

NS_PER_SEC = 1_000_000_000
_SENTINEL = object()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


@dataclass
class Config:
    # Which model AirLLM loads. Defaults to the intended large local model;
    # override this when you want a smaller smoke-test model.
    model_id: str = field(
        default_factory=lambda: os.getenv("AIRLLM_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
    )
    # The name clients see. Defaults to model_id; set it if you want a
    # "qwen2.5:72b"-style entry in Open WebUI's model picker.
    model_alias: str = field(default_factory=lambda: os.getenv("AIRLLM_MODEL_ALIAS", ""))

    # AirLLMBaseModel.__init__ defaults to device="cuda:0", so for RAM-based
    # layer swapping this MUST be passed explicitly.
    device: str = field(default_factory=lambda: os.getenv("AIRLLM_DEVICE", "cpu"))
    # "4bit"/"8bit" need `pip install bitsandbytes` and are not dependable on
    # CPU-only hosts. Empty means no compression.
    compression: str = field(default_factory=lambda: os.getenv("AIRLLM_COMPRESSION", ""))
    # AirLLM's own default is 512, which silently truncates longer prompts.
    max_seq_len: int = field(default_factory=lambda: _env_int("AIRLLM_MAX_SEQ_LEN", 2048))
    layer_shards_path: str = field(
        default_factory=lambda: os.getenv("AIRLLM_LAYER_SHARDS_PATH", "")
    )
    prefetching: bool = field(default_factory=lambda: _env_bool("AIRLLM_PREFETCHING", True))
    hf_token: str = field(
        default_factory=lambda: os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_HUB_TOKEN", ""))
    )

    max_new_tokens: int = field(default_factory=lambda: _env_int("AIRLLM_MAX_NEW_TOKENS", 256))
    # A single layer-swapped token can legitimately take minutes, so the
    # streamer timeout is generous.
    token_timeout: int = field(default_factory=lambda: _env_int("AIRLLM_TOKEN_TIMEOUT", 1800))

    host: str = field(default_factory=lambda: os.getenv("AIRLLM_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("AIRLLM_PORT", 11434))

    # Start loading weights at boot rather than on first request. The HTTP
    # port binds immediately either way.
    eager_load: bool = field(default_factory=lambda: _env_bool("AIRLLM_EAGER_LOAD", True))
    # How long a request waits for a still-loading model before 503.
    load_wait: int = field(default_factory=lambda: _env_int("AIRLLM_LOAD_WAIT", 30))
    # Version reported to clients. Open WebUI feature-gates on this.
    ollama_version: str = field(
        default_factory=lambda: os.getenv("AIRLLM_OLLAMA_VERSION", "0.32.15")
    )

    @property
    def public_name(self) -> str:
        return self.model_alias or self.model_id


cfg = Config()


# --------------------------------------------------------------------------
# Ollama wire format
# --------------------------------------------------------------------------


def now_iso() -> str:
    """RFC3339 with nanosecond precision and a Z suffix, as Ollama emits."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f000Z")


def fake_digest(text: str) -> str:
    """Clients display a digest; nothing verifies it against real blobs."""
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def parameter_size(model_id: str) -> str:
    """Best-effort '70B'-style label parsed out of the repo name."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", re.sub(r"[-_]", " ", model_id))
    return f"{match.group(1)}B" if match else "unknown"


def model_details() -> Dict[str, Any]:
    return {
        "parent_model": "",
        # Deliberately not "gguf" — AirLLM serves safetensors shards.
        "format": "safetensors",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": parameter_size(cfg.model_id),
        "quantization_level": cfg.compression or "F16",
    }


def model_entry() -> Dict[str, Any]:
    return {
        "name": cfg.public_name,
        "model": cfg.public_name,
        "modified_at": now_iso(),
        "size": 0,
        "digest": fake_digest(cfg.model_id),
        "details": model_details(),
    }


@dataclass
class Timing:
    """The nanosecond counters Ollama clients read to build usage stats."""

    started: float = field(default_factory=time.perf_counter)
    load_ns: int = 0
    prompt_eval_ns: int = 0
    prompt_tokens: int = 0
    eval_ns: int = 0
    eval_tokens: int = 0

    def as_packet_fields(self) -> Dict[str, Any]:
        return {
            "total_duration": int((time.perf_counter() - self.started) * NS_PER_SEC),
            "load_duration": self.load_ns,
            "prompt_eval_count": self.prompt_tokens,
            "prompt_eval_duration": self.prompt_eval_ns,
            "eval_count": self.eval_tokens,
            "eval_duration": self.eval_ns,
        }


def chat_chunk(content: str) -> Dict[str, Any]:
    return {
        "model": cfg.public_name,
        "created_at": now_iso(),
        "message": {"role": "assistant", "content": content, "images": None},
        "done": False,
    }


def chat_final(timing: Timing, done_reason: str = "stop") -> Dict[str, Any]:
    # LiteLLM requires `message` on every packet, the final one included, and
    # reads usage from prompt_eval_count / eval_count.
    return {
        "model": cfg.public_name,
        "created_at": now_iso(),
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": done_reason,
        **timing.as_packet_fields(),
    }


def generate_chunk(content: str) -> Dict[str, Any]:
    return {
        "model": cfg.public_name,
        "created_at": now_iso(),
        "response": content,
        "done": False,
    }


def generate_final(timing: Timing, done_reason: str = "stop") -> Dict[str, Any]:
    return {
        "model": cfg.public_name,
        "created_at": now_iso(),
        "response": "",
        "done": True,
        "done_reason": done_reason,
        "context": [],
        **timing.as_packet_fields(),
    }


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


class ModelNotReady(RuntimeError):
    pass


class Generation:
    """A single in-flight generation.

    The producer thread owns the whole lifecycle: it takes the model lock,
    runs the model, and releases the lock only once generation has actually
    stopped. The HTTP layer only ever reads from the queue and may set
    `cancel`, so a disconnecting client can never block the event loop or
    release the lock out from under a still-running generation.
    """

    def __init__(self) -> None:
        self.chunks: "queue.Queue[Any]" = queue.Queue(maxsize=256)
        self.cancel = threading.Event()
        self.timing = Timing()
        self.error: Optional[BaseException] = None

    def collect(self) -> str:
        """Block until generation finishes; return the full text."""
        parts: List[str] = []
        while True:
            item = self.chunks.get()
            if item is _SENTINEL:
                break
            parts.append(item)
        if self.error is not None:
            raise self.error
        return "".join(parts)


class ModelManager:
    """Owns the AirLLM model and serializes access to it.

    AirLLM swaps layers through one shared set of buffers, so exactly one
    generation may be in flight at a time.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.model: Any = None
        self.load_error: Optional[BaseException] = None
        self.load_ns = 0
        self._ready = threading.Event()
        self._gen_lock = threading.Lock()
        self._load_lock = threading.Lock()

    # -- loading ----------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def status(self) -> str:
        if self.ready:
            return "ready"
        return "error" if self.load_error is not None else "loading"

    def load(self) -> None:
        with self._load_lock:
            if self._ready.is_set():
                return
            started = time.perf_counter()
            try:
                from airllm import AutoModel  # late import keeps startup cheap

                kwargs: Dict[str, Any] = {
                    "device": self.cfg.device,
                    "prefetching": self.cfg.prefetching,
                }
                if self.cfg.max_seq_len:
                    kwargs["max_seq_len"] = self.cfg.max_seq_len
                if self.cfg.compression:
                    kwargs["compression"] = self.cfg.compression
                if self.cfg.layer_shards_path:
                    kwargs["layer_shards_saving_path"] = self.cfg.layer_shards_path
                if self.cfg.hf_token:
                    kwargs["hf_token"] = self.cfg.hf_token

                log.info(
                    "loading %s on %s (compression=%s, max_seq_len=%s) — the first "
                    "run downloads and shards the weights, so expect this to be slow",
                    self.cfg.model_id,
                    self.cfg.device,
                    self.cfg.compression or "none",
                    self.cfg.max_seq_len,
                )
                self.model = AutoModel.from_pretrained(self.cfg.model_id, **kwargs)
                self.load_ns = int((time.perf_counter() - started) * NS_PER_SEC)
                self.load_error = None
                self._ready.set()
                log.info("model ready after %.1fs", self.load_ns / NS_PER_SEC)
            except BaseException as exc:  # noqa: BLE001 - surfaced via /health
                self.load_error = exc
                log.exception("model load failed")
                raise

    def load_in_background(self) -> None:
        def run() -> None:
            try:
                self.load()
            except BaseException:  # noqa: BLE001 - already logged and recorded
                pass

        threading.Thread(target=run, name="airllm-load", daemon=True).start()

    def require(self, timeout: Optional[float] = None) -> None:
        """Block until the model is usable, or raise ModelNotReady."""
        if self.ready:
            return
        if not self.cfg.eager_load:
            self.load_in_background()
        wait = self.cfg.load_wait if timeout is None else timeout
        if not self._ready.wait(timeout=wait):
            if self.load_error is not None:
                raise ModelNotReady(f"model failed to load: {self.load_error}")
            raise ModelNotReady(
                "model is still loading; check /health or "
                "`journalctl -u airllm-ollama-api -f` for progress"
            )

    # -- prompting --------------------------------------------------------

    def tokenize(self, text: str):
        return self.model.tokenizer(
            text,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=self.cfg.max_seq_len,
            padding=False,
        )

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.model.tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001 - usage numbers are advisory
            return 0

    def build_prompt(self, messages: List["ChatMessage"]) -> str:
        """Prefer the tokenizer's own chat template; fall back to Llama 3."""
        payload = [{"role": m.role, "content": m.content} for m in messages]
        tokenizer = self.model.tokenizer
        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    payload, tokenize=False, add_generation_prompt=True
                )
            except Exception:  # noqa: BLE001 - fall through to the manual template
                log.warning("apply_chat_template failed; using the manual Llama 3 template")
        return llama3_prompt(payload)

    # -- generation -------------------------------------------------------

    def submit(
        self, prompt: str, max_new_tokens: int, gen_options: Dict[str, Any]
    ) -> Generation:
        """Start a generation on a background thread and return its handle."""
        gen = Generation()

        def producer() -> None:
            try:
                # Queue behind any generation already using the model.
                with self._gen_lock:
                    self._run(prompt, gen, max_new_tokens, gen_options)
            except BaseException as exc:  # noqa: BLE001 - re-raised to the caller
                gen.error = exc
                log.exception("generation failed")
            finally:
                gen.chunks.put(_SENTINEL)

        threading.Thread(target=producer, name="airllm-producer", daemon=True).start()
        return gen

    def _run(
        self,
        prompt: str,
        gen: Generation,
        max_new_tokens: int,
        gen_options: Dict[str, Any],
    ) -> None:
        from transformers import (
            StoppingCriteria,
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

        class CancelCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: D102
                return gen.cancel.is_set()

        gen.timing.load_ns = self.load_ns
        tokenizer = self.model.tokenizer

        tokenize_started = time.perf_counter()
        inputs = self.tokenize(prompt)
        input_ids = inputs["input_ids"]
        gen.timing.prompt_tokens = int(input_ids.shape[-1])
        gen.timing.prompt_eval_ns = int((time.perf_counter() - tokenize_started) * NS_PER_SEC)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=self.cfg.token_timeout,
        )
        kwargs: Dict[str, Any] = dict(
            inputs=input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            # Without a KV cache, every token replays every layer from disk.
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([CancelCriteria()]),
            **gen_options,
        )

        failure: List[BaseException] = []

        def run_model() -> None:
            try:
                self.model.generate(**kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised by the producer
                failure.append(exc)
            finally:
                # Always unblock the iterator, or the request hangs forever.
                try:
                    streamer.end()
                except Exception:  # noqa: BLE001
                    pass

        worker = threading.Thread(target=run_model, name="airllm-generate", daemon=True)
        eval_started = time.perf_counter()
        worker.start()

        produced: List[str] = []
        try:
            for text in streamer:
                if not text:
                    continue
                produced.append(text)
                if gen.cancel.is_set():
                    break
                try:
                    gen.chunks.put(text, timeout=self.cfg.token_timeout)
                except queue.Full:  # consumer vanished
                    gen.cancel.set()
                    break
        finally:
            # Safe to block here: this is the producer thread, not the loop.
            worker.join(timeout=self.cfg.token_timeout)
            gen.timing.eval_ns = int((time.perf_counter() - eval_started) * NS_PER_SEC)
            gen.timing.eval_tokens = self.count_tokens("".join(produced))

        if failure:
            raise failure[0]


def llama3_prompt(messages: List[Dict[str, str]]) -> str:
    """Manual Llama 3 template, used only when the tokenizer has none.

    Roles are lowercase and the sequence opens with <|begin_of_text|>.
    """
    out = "<|begin_of_text|>"
    for msg in messages:
        role = str(msg.get("role", "user")).lower()
        out += (
            f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            f"{msg.get('content', '')}<|eot_id|>"
        )
    out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return out


manager = ModelManager(cfg)


# --------------------------------------------------------------------------
# Request schemas
# --------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""
    images: Optional[List[str]] = None


class GenerateRequest(BaseModel):
    model: Optional[str] = None
    prompt: str = ""
    system: Optional[str] = None
    stream: bool = True
    raw: bool = False
    options: Dict[str, Any] = Field(default_factory=dict)
    keep_alive: Optional[Any] = None


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    stream: bool = True
    options: Dict[str, Any] = Field(default_factory=dict)
    keep_alive: Optional[Any] = None
    tools: Optional[List[Dict[str, Any]]] = None


class ShowRequest(BaseModel):
    model: Optional[str] = None
    name: Optional[str] = None


class ModelRequest(BaseModel):
    model: Optional[str] = None
    name: Optional[str] = None
    stream: bool = True


def model_names() -> set:
    names = {cfg.model_id, cfg.public_name}
    names.update(f"{name}:latest" for name in list(names) if ":" not in name)
    return {name for name in names if name}


def requested_model_name(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def assert_known_model(*values: Optional[str]) -> None:
    requested = requested_model_name(*values)
    if requested is None:
        return
    if requested not in model_names():
        raise HTTPException(
            status_code=404,
            detail=f"model {requested!r} is not served by this AirLLM process",
        )


def resolve_options(options: Dict[str, Any]) -> tuple:
    """Translate Ollama `options` into transformers generate() kwargs."""
    max_new_tokens = cfg.max_new_tokens
    num_predict = options.get("num_predict")
    if isinstance(num_predict, int) and num_predict > 0:
        max_new_tokens = num_predict

    gen: Dict[str, Any] = {}
    temperature = options.get("temperature")
    if isinstance(temperature, (int, float)):
        if temperature > 0:
            gen["do_sample"] = True
            gen["temperature"] = float(temperature)
        else:
            gen["do_sample"] = False
    for src, dst in (
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("repeat_penalty", "repetition_penalty"),
    ):
        value = options.get(src)
        if isinstance(value, (int, float)):
            gen[dst] = value
    seed = options.get("seed")
    if isinstance(seed, int) and seed > 0:
        try:
            import torch

            torch.manual_seed(seed)
        except Exception:  # noqa: BLE001
            pass
    return max_new_tokens, gen


def prompt_for_generate(req: GenerateRequest) -> str:
    if req.raw:
        return req.prompt
    messages: List[ChatMessage] = []
    if req.system:
        messages.append(ChatMessage(role="system", content=req.system))
    messages.append(ChatMessage(role="user", content=req.prompt))
    return manager.build_prompt(messages)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    if cfg.eager_load:
        # Background thread: the socket binds now and /api/version, /api/tags
        # answer immediately, so clients can connect while weights load.
        manager.load_in_background()
    yield


app = FastAPI(title="AirLLM Ollama-compatible API", lifespan=lifespan)


def ndjson_stream(
    request: Request,
    gen: Generation,
    chunk_fn: Callable[[str], Dict[str, Any]],
    final_fn: Callable[[Timing], Dict[str, Any]],
) -> StreamingResponse:
    """Drain a Generation to the client as NDJSON, honouring disconnects."""

    async def body() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        get = functools.partial(gen.chunks.get, timeout=1.0)
        finished = False
        try:
            while True:
                try:
                    item = await loop.run_in_executor(None, get)
                except queue.Empty:
                    # No token yet — a good moment to notice a vanished client.
                    if await request.is_disconnected():
                        return
                    continue
                if item is _SENTINEL:
                    break
                yield json.dumps(chunk_fn(item)) + "\n"
                if await request.is_disconnected():
                    return
            if gen.error is not None:
                yield json.dumps(
                    {"error": f"{type(gen.error).__name__}: {gen.error}"}
                ) + "\n"
            else:
                yield json.dumps(final_fn(gen.timing)) + "\n"
            finished = True
        finally:
            # Starlette watches for http.disconnect itself and cancels this
            # generator, so `is_disconnected()` above often never sees it.
            # This is the branch that actually stops a runaway generation and
            # frees the model lock when a client hangs up.
            if not finished:
                log.info("stream ended early; cancelling generation")
                gen.cancel.set()

    return StreamingResponse(body(), media_type="application/x-ndjson")


async def run_blocking(fn):
    """Run a blocking call off the event loop and map errors onto HTTP."""
    try:
        return await asyncio.get_running_loop().run_in_executor(None, fn)
    except ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("request failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/", response_class=PlainTextResponse)
@app.head("/", response_class=PlainTextResponse)
async def root() -> str:
    return "Ollama is running"


@app.get("/api/version")
async def version() -> Dict[str, str]:
    return {"version": cfg.ollama_version}


@app.get("/api/tags")
async def tags() -> Dict[str, Any]:
    return {"models": [model_entry()]}


@app.get("/api/ps")
async def ps() -> Dict[str, Any]:
    if not manager.ready:
        return {"models": []}
    entry = model_entry()
    entry.pop("modified_at", None)
    entry["expires_at"] = "0001-01-01T00:00:00Z"  # never unloaded
    entry["size_vram"] = 0  # layers live in RAM, not VRAM
    return {"models": [entry]}


@app.post("/api/show")
async def show(req: ShowRequest) -> Dict[str, Any]:
    assert_known_model(req.model, req.name)
    return {
        "modelfile": f"FROM {cfg.model_id}",
        "parameters": f"num_ctx {cfg.max_seq_len}\nnum_predict {cfg.max_new_tokens}",
        "template": "{{ .Prompt }}",
        "details": model_details(),
        "model_info": {
            "general.architecture": "llama",
            "general.parameter_count": 0,
            "llama.context_length": cfg.max_seq_len,
        },
        "capabilities": ["completion"],
    }


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    assert_known_model(req.model)
    if req.tools:
        log.warning("tool calling is unsupported; ignoring %d tool definitions", len(req.tools))

    def start() -> Generation:
        manager.require()
        prompt = manager.build_prompt(req.messages)
        max_new_tokens, gen_options = resolve_options(req.options)
        return manager.submit(prompt, max_new_tokens, gen_options)

    gen = await run_blocking(start)

    if req.stream:
        return ndjson_stream(request, gen, chat_chunk, chat_final)

    def finish() -> Dict[str, Any]:
        text = gen.collect()
        packet = chat_final(gen.timing)
        packet["message"] = {"role": "assistant", "content": text}
        return packet

    return await run_blocking(finish)


@app.post("/api/generate")
async def generate(req: GenerateRequest, request: Request):
    assert_known_model(req.model)

    def start() -> Generation:
        manager.require()
        prompt = prompt_for_generate(req)
        max_new_tokens, gen_options = resolve_options(req.options)
        return manager.submit(prompt, max_new_tokens, gen_options)

    gen = await run_blocking(start)

    if req.stream:
        return ndjson_stream(request, gen, generate_chunk, generate_final)

    def finish() -> Dict[str, Any]:
        text = gen.collect()
        packet = generate_final(gen.timing)
        packet["response"] = text
        return packet

    return await run_blocking(finish)


@app.post("/api/embeddings")
@app.post("/api/embed")
async def embeddings() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"error": "embeddings are not supported by the AirLLM wrapper"},
    )


@app.post("/api/pull")
async def pull(req: ModelRequest) -> Any:
    assert_known_model(req.model, req.name)
    if not manager.ready:
        manager.load_in_background()

    if not req.stream:
        return {"status": "success"}

    async def body() -> AsyncIterator[str]:
        for status in ("pulling manifest", "verifying sha256 digest", "success"):
            yield json.dumps({"status": status}) + "\n"

    return StreamingResponse(body(), media_type="application/x-ndjson")


@app.post("/api/create")
@app.post("/api/copy")
@app.post("/api/push")
async def unsupported_model_write() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": "this AirLLM wrapper serves one configured Hugging Face model; "
            "Ollama model creation, copying, and pushing are not supported"
        },
    )


@app.delete("/api/delete")
@app.post("/api/delete")
async def delete(req: ModelRequest) -> JSONResponse:
    assert_known_model(req.model, req.name)
    return JSONResponse(
        status_code=501,
        content={"error": "the configured AirLLM model cannot be deleted through this API"},
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": manager.status(),
        "model": cfg.public_name,
        "model_id": cfg.model_id,
        "device": cfg.device,
        "compression": cfg.compression or None,
        "max_seq_len": cfg.max_seq_len,
        "error": None if manager.load_error is None else str(manager.load_error),
    }


def main() -> None:
    import uvicorn

    if cfg.host == "0.0.0.0":  # noqa: S104 - warn, don't override
        log.warning(
            "binding 0.0.0.0: this endpoint has no authentication and will be "
            "reachable by anything on your network"
        )
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
