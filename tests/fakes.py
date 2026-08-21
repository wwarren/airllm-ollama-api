"""Stand-ins for airllm and transformers so the wrapper can be tested fast.

Both are injected into sys.modules before airllm_ollama_api.py is imported. That only
works because airllm_ollama_api.py imports them lazily inside functions — if someone
moves those imports to module scope, these tests fail loudly, which is the
point.
"""

from __future__ import annotations

import queue
import sys
import threading
import types
from typing import Any, Dict, List, Optional


class FakeTensor:
    """Just enough of a torch tensor for `input_ids.shape[-1]`."""

    def __init__(self, ids: List[int]) -> None:
        self.ids = ids
        self.shape = (1, len(ids))


class FakeTokenizer:
    def __init__(self, chat_template: Optional[str] = "present") -> None:
        self.chat_template = chat_template
        self.last_call: Dict[str, Any] = {}

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        self.last_call = {"text": text, **kwargs}
        ids = self.encode(text)
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length:
            ids = ids[:max_length]
        return {"input_ids": FakeTensor(ids)}

    def encode(self, text: str, **_kwargs) -> List[int]:
        return [len(word) for word in text.split()] or ([0] if text else [])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False) -> str:
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


class FakeModel:
    """Emits a fixed reply one word at a time, honouring the stop criteria."""

    REPLY = ["Hello", " there", " from", " AirLLM", "!"]

    def __init__(self, tokenizer: FakeTokenizer, fail_with: Optional[Exception] = None) -> None:
        self.tokenizer = tokenizer
        self.fail_with = fail_with
        self.calls: List[Dict[str, Any]] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self.delay = 0.0
        self._lock = threading.Lock()

    def generate(self, **kwargs) -> None:
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            self.calls.append(kwargs)
            if self.fail_with is not None:
                raise self.fail_with
            streamer = kwargs["streamer"]
            criteria = kwargs.get("stopping_criteria") or []
            budget = kwargs.get("max_new_tokens", 256)
            for index, token in enumerate(self.REPLY):
                if index >= budget:
                    break
                if any(check(None, None) for check in criteria):
                    break
                if self.delay:
                    threading.Event().wait(self.delay)
                streamer.put_text(token)
        finally:
            with self._lock:
                self.concurrent -= 1


class FakeTextIteratorStreamer:
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.timeout = timeout
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._stop = object()

    def put_text(self, text: str) -> None:
        self._queue.put(text)

    def end(self) -> None:
        self._queue.put(self._stop)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get(timeout=self.timeout)
        if item is self._stop:
            raise StopIteration
        return item


class FakeStoppingCriteria:
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return False


class FakeStoppingCriteriaList(list):
    pass


def install(model: FakeModel) -> None:
    """Register the fake airllm and transformers modules."""
    airllm = types.ModuleType("airllm")

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            AutoModel.last_kwargs = dict(kwargs)
            AutoModel.last_model_id = model_id
            return model

    AutoModel.last_kwargs = {}
    AutoModel.last_model_id = ""
    airllm.AutoModel = AutoModel
    sys.modules["airllm"] = airllm

    transformers = types.ModuleType("transformers")
    transformers.TextIteratorStreamer = FakeTextIteratorStreamer
    transformers.StoppingCriteria = FakeStoppingCriteria
    transformers.StoppingCriteriaList = FakeStoppingCriteriaList
    sys.modules["transformers"] = transformers
