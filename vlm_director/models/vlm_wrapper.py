"""Minimal backend contract with Qwen, compatible HTTP, and explicit fixture replay.

The HTTP backend extends the pattern in the existing llm_provider.py. That module
imports the generator and writes llm_events.json, so it is intentionally not imported.
Optional ML dependencies are loaded only when real local inference is requested.
"""

from __future__ import annotations

import base64
import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..prompts.director_prompt import SYSTEM_PROMPT

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


class ModelError(RuntimeError):
    """Model setup, inference, or compatible API transport failed."""


@dataclass(frozen=True)
class GenerationSettings:
    """Generation controls are recorded with every experiment for reproducibility."""

    seed: int = 0
    temperature: float = 0.0
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        """Reject invalid controls before allocating memory or issuing requests."""
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2^32)")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and >= 0")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")


class VLMWrapper(Protocol):
    """New backends implement raw text generation and descriptive metadata only."""

    supports_vision: bool

    @property
    def metadata(self) -> dict[str, Any]:
        """Identify the backend and distinguish fixture data from model inference."""
        ...

    def generate(self, prompt: str, image_path: Path | None,
                 settings: GenerationSettings) -> str:
        """Return raw assistant text; parsing and feasibility stay in the Director."""
        ...


def read_bev(path: str | Path) -> tuple[Path, bytes, str]:
    """Validate a local PNG/JPEG input with no third-party dependency.

    Signature checks prevent sending arbitrary files as images. Real image decoding
    remains the processor/server's responsibility; BEV semantics require the user
    to provide a snapshot-aligned image and optional calibration metadata.
    """
    image_path = Path(path).resolve()
    if not image_path.is_file():
        raise ModelError(f"BEV image does not exist: {image_path}")
    if image_path.stat().st_size > 20 * 1024 * 1024:
        raise ModelError("BEV image exceeds the 20 MiB baseline limit")
    data = image_path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    else:
        raise ModelError("BEV must be a PNG or JPEG image")
    return image_path, data, mime


class QwenVLMWrapper:
    """Local Qwen2.5-VL baseline using the documented Transformers 4.57 interface."""

    supports_vision = True

    def __init__(self, model_id: str = DEFAULT_QWEN_MODEL, *, device: str = "auto",
                 local_files_only: bool = False) -> None:
        """Store configuration; defer weight loading until the first generation."""
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        self.model_id, self.device = model_id, device
        self.local_files_only = local_files_only
        self._model = self._processor = self._torch = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return only reproducibility information, never credentials."""
        return {"backend": "transformers", "model_id": self.model_id,
                "device": self.device, "local_files_only": self.local_files_only,
                "is_mock": False}

    def _load(self) -> None:
        """Load one model on one device, with no remote Python code or training."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ModelError(
                "Local Qwen requires optional dependencies. Run: "
                "python -m pip install -r vlm_director/requirements-qwen.txt; "
                "or use --backend api --base-url http://localhost:8000/v1.") from exc
        device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ModelError("CUDA requested but unavailable")
        dtype = torch.float32 if device == "cpu" else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        try:
            processor = AutoProcessor.from_pretrained(
                self.model_id, local_files_only=self.local_files_only,
                trust_remote_code=False, max_pixels=512 * 28 * 28)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, local_files_only=self.local_files_only,
                trust_remote_code=False, dtype=dtype, attn_implementation="sdpa")
            model = model.to(device).eval()
        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelError(f"Could not load Qwen checkpoint: {exc}") from exc
        self.device = device
        self._torch, self._processor, self._model = torch, processor, model

    def generate(self, prompt: str, image_path: Path | None,
                 settings: GenerationSettings) -> str:
        """Use the same Qwen checkpoint for text and BEV ablations."""
        self._load()
        content: list[dict[str, Any]] = []
        if image_path is not None:
            image_path, _, _ = read_bev(image_path)
            content.append({"type": "image", "path": str(image_path)})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}]
        try:
            inputs = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(self._model.device)
            options: dict[str, Any] = {"max_new_tokens": settings.max_new_tokens,
                                       "do_sample": settings.temperature > 0}
            if settings.temperature > 0:
                options["temperature"] = settings.temperature
            # Preserve the caller's RNG state while making each inference repeatable.
            devices = [self._torch.cuda.current_device()] if self.device == "cuda" else []
            with self._torch.random.fork_rng(devices=devices):
                self._torch.manual_seed(settings.seed)
                with self._torch.inference_mode():
                    ids = self._model.generate(**inputs, **options)
            # Decode completion tokens only; including the prompt would corrupt parsing.
            trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, ids)]
            return self._processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelError(f"Qwen inference failed: {exc}") from exc


class CompatibleAPIWrapper:
    """Stdlib-only text/image client for a user-configured compatible model server."""

    supports_vision = True

    def __init__(self, base_url: str, model_id: str, *, timeout: float = 90,
                 json_mode: bool = True) -> None:
        """Follow the repository's LLM_API_KEY convention; never store it in reports."""
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not model_id or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("model_id must be nonempty and timeout must be positive")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model_id, self.timeout, self.json_mode = model_id, timeout, json_mode
        self._effective_json_mode = json_mode

    @property
    def metadata(self) -> dict[str, Any]:
        """Record actual JSON mode after any capability-specific fallback."""
        return {"backend": "api", "model_id": self.model_id, "is_mock": False,
                "json_mode": self._effective_json_mode}

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Issue a bounded request; HTTP errors are handled by generate()."""
        headers = {"Content-Type": "application/json"}
        if os.environ.get("LLM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["LLM_API_KEY"]
        request = urllib.request.Request(
            self.url, data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def generate(self, prompt: str, image_path: Path | None,
                 settings: GenerationSettings) -> str:
        """Actually transmit BEV bytes, rather than passing a local path as text."""
        content: str | list[dict[str, Any]] = prompt
        if image_path is not None:
            _, data, mime = read_bev(image_path)
            data_url = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": data_url}}]
        payload: dict[str, Any] = {
            "model": self.model_id, "temperature": settings.temperature,
            "max_tokens": settings.max_new_tokens, "seed": settings.seed,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": content}],
        }
        self._effective_json_mode = self.json_mode
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            try:
                response = self._request(payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read(8192).decode("utf-8", errors="replace").lower()
                if (exc.code not in (400, 422) or "response_format" not in payload
                        or not any(word in detail for word in ("response_format", "json_object"))):
                    raise ModelError(f"Compatible API returned HTTP {exc.code}") from exc
                # Retry only a reported unsupported JSON-mode capability, not any 400.
                payload.pop("response_format")
                self._effective_json_mode = False
                response = self._request(payload)
            result = response["choices"][0]["message"]["content"]
            if isinstance(result, list):
                result = "".join(part["text"] for part in result if part.get("type") == "text")
            if not isinstance(result, str) or not result.strip():
                raise ModelError("Compatible API returned empty/non-text content")
            return result
        except urllib.error.HTTPError as exc:
            raise ModelError(f"Compatible API returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelError(f"Compatible API connection failed: {type(exc).__name__}") from exc
        except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise ModelError("Compatible API returned an invalid response envelope") from exc


class ReplayWrapper:
    """Explicit fixture backend for pipeline tests; it performs no VLM inference."""

    supports_vision = True

    def __init__(self, response_file: str | Path) -> None:
        """A file is required so fixture behavior can never be an implicit fallback."""
        self.response_file = Path(response_file)

    @property
    def metadata(self) -> dict[str, Any]:
        """Mark replay output as mock throughout reports and evaluation."""
        return {"backend": "replay", "model_id": "fixture", "is_mock": True}

    def generate(self, prompt: str, image_path: Path | None,
                 settings: GenerationSettings) -> str:
        """Read recorded raw text; deliberately ignore prompt/image content."""
        return self.response_file.read_text(encoding="utf-8-sig")
