"""llama.cpp server backend for the sermon EuroLLM GGUF.

The server is expected to expose llama.cpp's ``/completion`` endpoint with the
ChatML prompt used during training. Start it with the command in
``release/sermon-model/README.md``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence

from ..errors import BackendError

PROMPT = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\n"
    "Translate the following Danish source text to English:\n"
    "Danish: {source}\n"
    "English: <|im_end|>\n"
    "<|im_start|>assistant\n"
)


class LlamaServerBackend:
    """HTTP client for a local llama.cpp translation server."""

    name = "llama"

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:18080",
        *,
        max_tokens: int = 256,
        timeout: float = 120.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"LlamaServerBackend(server_url={self.server_url!r})"

    def _translate_one(self, source: str) -> str:
        body = {
            "prompt": PROMPT.format(source=source),
            "n_predict": self.max_tokens,
            "temperature": 0,
            "stop": ["<|im_end|>"],
        }
        request = urllib.request.Request(
            f"{self.server_url}/completion",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise BackendError(
                f"llama.cpp server request failed at {self.server_url}: {exc}"
            ) from exc
        return str(data.get("content", "")).strip()

    def translate(self, texts: Sequence[str], *, beam_size: int | None = None) -> list[str]:
        if isinstance(texts, str):  # defensive: the protocol is sequence-based
            texts = [texts]
        results: list[str] = []
        for text in texts:
            source = " ".join(text.split())
            results.append(self._translate_one(source) if source else "")
        return results

    def warmup(self) -> None:
        self.translate(["Gud er kærlighed."])

    def close(self) -> None:
        """The HTTP server is owned by the caller; nothing to release."""
