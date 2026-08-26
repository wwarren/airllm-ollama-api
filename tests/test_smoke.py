"""End-to-end checks against a stubbed model.

Run with:  pytest -q
No weights are downloaded; airllm and transformers are faked (see fakes.py).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fakes  # noqa: E402


def build(fail_with=None, chat_template="present", **env):
    """Fresh fakes + a fresh server module with the given environment."""
    defaults = {
        "AIRLLM_MODEL_ID": "fake/Test-7B-Instruct",
        "AIRLLM_DEVICE": "cpu",
        "AIRLLM_MAX_SEQ_LEN": "128",
        "AIRLLM_MAX_NEW_TOKENS": "64",
        "AIRLLM_LOAD_WAIT": "10",
        "AIRLLM_TOKEN_TIMEOUT": "10",
        "AIRLLM_HOST": "127.0.0.1",
        "AIRLLM_PORT": "11434",
    }
    defaults.update({k: str(v) for k, v in env.items()})
    for key in list(os.environ):
        if key.startswith("AIRLLM_"):
            del os.environ[key]
    os.environ.update(defaults)

    tokenizer = fakes.FakeTokenizer(chat_template=chat_template)
    model = fakes.FakeModel(tokenizer, fail_with=fail_with)
    fakes.install(model)

    # Plain import now that the module name is a valid identifier.
    sys.modules.pop("airllm_ollama_api", None)
    server = importlib.import_module("airllm_ollama_api")
    return server, model


def ndjson(response):
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Metadata endpoints — what clients probe on connect
# --------------------------------------------------------------------------


def test_metadata_endpoints_answer_before_the_model_loads():
    server, _ = build(AIRLLM_EAGER_LOAD="false")
    with TestClient(server.app) as client:
        assert client.get("/").text == "Ollama is running"
        assert client.get("/api/version").json() == {"version": server.cfg.ollama_version}

        tags = client.get("/api/tags").json()
        entry = tags["models"][0]
        for key in ("name", "model", "modified_at", "size", "digest", "details"):
            assert key in entry, f"/api/tags entry is missing {key}"
        # AirLLM serves safetensors shards, never gguf.
        assert entry["details"]["format"] == "safetensors"
        assert entry["details"]["parameter_size"] == "7B"

        show = client.post("/api/show", json={"model": "fake/Test-7B-Instruct"}).json()
        for key in ("modelfile", "parameters", "template", "details", "capabilities"):
            assert key in show, f"/api/show is missing {key}"

        # Nothing is loaded yet, so /api/ps is empty.
        assert client.get("/api/ps").json() == {"models": []}
        assert client.get("/health").json()["status"] == "loading"


def test_ps_and_health_report_a_loaded_model():
    server, _ = build()
    with TestClient(server.app) as client:
        server.manager.require()
        assert client.get("/health").json()["status"] == "ready"
        entry = client.get("/api/ps").json()["models"][0]
        assert entry["size_vram"] == 0  # layers live in RAM


def test_embeddings_are_refused_clearly():
    server, _ = build()
    with TestClient(server.app) as client:
        assert client.post("/api/embed", json={"input": "x"}).status_code == 501


def test_unknown_model_requests_are_rejected():
    server, _ = build(AIRLLM_EAGER_LOAD="false")
    with TestClient(server.app) as client:
        response = client.post(
            "/api/generate",
            json={"model": "not-the-configured-model", "prompt": "hi", "stream": False},
        )
    assert response.status_code == 404
    assert "not-the-configured-model" in response.json()["detail"]


def test_model_alias_and_latest_tag_are_accepted():
    server, _ = build(AIRLLM_MODEL_ALIAS="local-airllm", AIRLLM_EAGER_LOAD="false")
    with TestClient(server.app) as client:
        assert client.post("/api/show", json={"model": "local-airllm"}).status_code == 200
        assert (
            client.post("/api/show", json={"model": "local-airllm:latest"}).status_code
            == 200
        )
        assert (
            client.post("/api/show", json={"model": "fake/Test-7B-Instruct"}).status_code
            == 200
        )


def test_pull_reports_success_for_the_configured_model_without_blocking_on_load():
    server, _ = build(AIRLLM_EAGER_LOAD="false")
    with TestClient(server.app) as client:
        packets = ndjson(client.post("/api/pull", json={"model": "fake/Test-7B-Instruct"}))
        once = client.post(
            "/api/pull", json={"model": "fake/Test-7B-Instruct", "stream": False}
        ).json()

    assert packets[-1] == {"status": "success"}
    assert once == {"status": "success"}


def test_unsupported_model_management_routes_are_explicit():
    server, _ = build(AIRLLM_EAGER_LOAD="false")
    with TestClient(server.app) as client:
        assert (
            client.post("/api/create", json={"model": "fake/Test-7B-Instruct"}).status_code
            == 501
        )
        assert (
            client.post("/api/copy", json={"source": "a", "destination": "b"}).status_code
            == 501
        )
        assert (
            client.post("/api/push", json={"model": "fake/Test-7B-Instruct"}).status_code
            == 501
        )
        assert (
            client.post("/api/delete", json={"model": "fake/Test-7B-Instruct"}).status_code
            == 501
        )
        assert client.request(
            "DELETE", "/api/delete", json={"model": "fake/Test-7B-Instruct"}
        ).status_code == 501


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


def test_chat_stream_packet_shapes():
    server, _ = build()
    with TestClient(server.app) as client:
        packets = ndjson(
            client.post(
                "/api/chat",
                json={"model": "fake/Test-7B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
            )
        )

    body, final = packets[:-1], packets[-1]
    assert "".join(p["message"]["content"] for p in body) == "".join(fakes.FakeModel.REPLY)

    for packet in packets:
        # LiteLLM requires model, message and done on every packet.
        assert packet["model"] and "created_at" in packet
        assert "message" in packet and "done" in packet
    assert all(p["done"] is False for p in body)

    assert final["done"] is True
    assert final["done_reason"] == "stop"
    assert final["message"] == {"role": "assistant", "content": ""}
    assert final["prompt_eval_count"] > 0
    assert final["eval_count"] > 0
    for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
        assert isinstance(final[key], int)


def test_chat_non_stream_matches_the_stream_and_never_echoes_the_prompt():
    server, _ = build()
    with TestClient(server.app) as client:
        payload = {
            "model": "fake/Test-7B-Instruct",
            "messages": [{"role": "user", "content": "unique-prompt-marker"}],
            "stream": False,
        }
        body = client.post("/api/chat", json=payload).json()

    assert body["done"] is True
    assert body["message"]["content"] == "".join(fakes.FakeModel.REPLY)
    assert "unique-prompt-marker" not in body["message"]["content"]
    assert body["eval_count"] > 0


def test_chat_uses_the_tokenizer_template_and_asks_for_a_generation_prompt():
    server, model = build()
    with TestClient(server.app) as client:
        client.post(
            "/api/chat",
            json={
                "model": "fake/Test-7B-Instruct",
                "messages": [
                    {"role": "system", "content": "be nice"},
                    {"role": "user", "content": "hi"},
                ],
            },
        )
    prompt = model.tokenizer.last_call["text"]
    assert prompt == "<system>be nice</system><user>hi</user><assistant>"


def test_manual_template_is_lowercase_and_has_a_bos_token():
    server, model = build(chat_template=None)
    with TestClient(server.app) as client:
        client.post(
            "/api/chat", json={"model": "fake/Test-7B-Instruct", "messages": [{"role": "USER", "content": "hi"}]}
        )
    prompt = model.tokenizer.last_call["text"]
    assert prompt.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "USER" not in prompt
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


# --------------------------------------------------------------------------
# Generate
# --------------------------------------------------------------------------


def test_generate_stream_and_non_stream():
    server, _ = build()
    with TestClient(server.app) as client:
        packets = ndjson(client.post("/api/generate", json={"model": "fake/Test-7B-Instruct", "prompt": "hi"}))
        once = client.post(
            "/api/generate", json={"model": "fake/Test-7B-Instruct", "prompt": "hi", "stream": False}
        ).json()

    assert "".join(p["response"] for p in packets[:-1]) == "".join(fakes.FakeModel.REPLY)
    assert packets[-1]["done"] is True and packets[-1]["response"] == ""
    assert packets[-1]["context"] == []
    assert once["response"] == "".join(fakes.FakeModel.REPLY)


def test_generate_raw_skips_templating():
    server, model = build()
    with TestClient(server.app) as client:
        client.post("/api/generate", json={"prompt": "bare prompt", "raw": True})
    assert model.tokenizer.last_call["text"] == "bare prompt"


# --------------------------------------------------------------------------
# Generation parameters and correctness guarantees
# --------------------------------------------------------------------------


def test_generation_always_uses_the_kv_cache():
    server, model = build()
    with TestClient(server.app) as client:
        client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    # Without use_cache every token replays every layer from disk.
    assert model.calls[0]["use_cache"] is True


def test_tokenizer_call_is_bounded_by_max_seq_len():
    server, model = build(AIRLLM_MAX_SEQ_LEN=64)
    with TestClient(server.app) as client:
        client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    call = model.tokenizer.last_call
    assert call["truncation"] is True
    assert call["max_length"] == 64
    assert call["return_attention_mask"] is False


def test_options_map_onto_generate_kwargs():
    server, model = build()
    with TestClient(server.app) as client:
        client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_predict": 2, "temperature": 0.7, "top_p": 0.9, "top_k": 40},
            },
        )
    call = model.calls[0]
    assert call["max_new_tokens"] == 2
    assert call["do_sample"] is True and call["temperature"] == 0.7
    assert call["top_p"] == 0.9 and call["top_k"] == 40


def test_num_predict_actually_truncates_the_reply():
    server, _ = build()
    with TestClient(server.app) as client:
        packets = ndjson(
            client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "options": {"num_predict": 2},
                },
            )
        )
    assert "".join(p["message"]["content"] for p in packets[:-1]) == "Hello there"


def test_unknown_client_fields_do_not_400():
    server, _ = build()
    with TestClient(server.app) as client:
        response = client.post(
            "/api/chat",
            json={
                "model": "fake/Test-7B-Instruct",
                "messages": [{"role": "user", "content": "hi", "images": None}],
                "keep_alive": "5m",
                "format": "json",
                "think": False,
            },
        )
    assert response.status_code == 200


def test_model_is_used_by_one_request_at_a_time():
    server, model = build()
    model.delay = 0.02
    results = []

    with TestClient(server.app) as client:
        server.manager.require()

        def fire():
            response = client.post(
                "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
            )
            results.append(ndjson(response))

        threads = [threading.Thread(target=fire) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

    assert len(results) == 3
    # AirLLM shares one set of layer buffers; overlapping generation corrupts it.
    assert model.max_concurrent == 1
    for packets in results:
        assert "".join(p["message"]["content"] for p in packets[:-1]) == "".join(
            fakes.FakeModel.REPLY
        )


# --------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------


def test_generation_failure_is_reported_instead_of_hanging():
    server, _ = build(fail_with=RuntimeError("out of memory"))
    with TestClient(server.app) as client:
        packets = ndjson(client.post("/api/chat", json={"messages": []}))
    assert "error" in packets[-1]
    assert "out of memory" in packets[-1]["error"]


def test_abandoning_a_stream_cancels_the_generation():
    """Starlette cancels the response generator on disconnect; that path has
    to stop the model, or the abandoned run holds the lock to completion."""
    server, _ = build()

    class NeverDisconnected:
        async def is_disconnected(self):
            return False

    gen = server.Generation()
    for token in ("a", "b", "c"):
        gen.chunks.put(token)

    response = server.ndjson_stream(
        NeverDisconnected(), gen, server.chat_chunk, server.chat_final
    )

    async def consume_one_then_hang_up():
        body = response.body_iterator
        await body.__anext__()
        await body.aclose()

    asyncio.run(consume_one_then_hang_up())
    assert gen.cancel.is_set()


def test_non_stream_generation_failure_returns_500():
    server, _ = build(fail_with=RuntimeError("boom"))
    with TestClient(server.app) as client:
        response = client.post("/api/chat", json={"messages": [], "stream": False})
    assert response.status_code == 500
    assert "boom" in response.json()["detail"]


def test_requests_503_while_the_model_is_still_loading():
    server, _ = build(AIRLLM_EAGER_LOAD="false", AIRLLM_LOAD_WAIT=0)

    def never_finishes() -> None:
        threading.Event().wait(30)

    server.manager.load = never_finishes  # type: ignore[assignment]
    with TestClient(server.app) as client:
        response = client.post("/api/chat", json={"messages": [], "stream": False})
    assert response.status_code == 503
    assert "loading" in response.json()["detail"]


def test_model_load_failure_surfaces_in_health():
    server, _ = build()

    def explode() -> None:
        raise RuntimeError("gated repo: set HF_TOKEN")

    server.manager.load = explode  # type: ignore[assignment]
    server.manager.load_error = RuntimeError("gated repo: set HF_TOKEN")
    with TestClient(server.app) as client:
        body = client.get("/health").json()
    assert body["status"] == "error"
    assert "HF_TOKEN" in body["error"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_airllm_is_constructed_with_an_explicit_device():
    server, _ = build(AIRLLM_MAX_SEQ_LEN=256, AIRLLM_COMPRESSION="")
    with TestClient(server.app):
        server.manager.require()
    kwargs = sys.modules["airllm"].AutoModel.last_kwargs
    # AirLLMBaseModel defaults to cuda:0; CPU swapping needs this set.
    assert kwargs["device"] == "cpu"
    assert kwargs["max_seq_len"] == 256
    assert "compression" not in kwargs  # empty means "don't pass it"


def test_hf_token_and_compression_are_forwarded_when_set():
    os.environ["HF_TOKEN"] = "hf_test"
    try:
        server, _ = build(AIRLLM_COMPRESSION="4bit")
        with TestClient(server.app):
            server.manager.require()
        kwargs = sys.modules["airllm"].AutoModel.last_kwargs
        assert kwargs["compression"] == "4bit"
        assert kwargs["hf_token"] == "hf_test"
    finally:
        del os.environ["HF_TOKEN"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
