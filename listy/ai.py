"""Thin wrapper around the Claude API for vision requests with structured output."""

from __future__ import annotations

import base64
import os
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

MODEL = os.environ.get("LISTY_MODEL", "claude-opus-5-5")

T = TypeVar("T", bound=BaseModel)


class RefusedError(RuntimeError):
    pass


class VisionModel(Protocol):
    def ask(self, parts: list[str | bytes], schema: type[T], *, system: str, effort: str = "medium") -> T: ...


class ClaudeVision:
    """Sends an interleaved list of text and JPEG bytes, returns a validated pydantic object."""

    def __init__(self, client: anthropic.Anthropic | None = None, model: str = MODEL):
        self.client = client or anthropic.Anthropic()
        self.model = model

    def ask(self, parts: list[str | bytes], schema: type[T], *, system: str, effort: str = "medium") -> T:
        content = []
        for part in parts:
            if isinstance(part, bytes):
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(part).decode("ascii"),
                    },
                })
            else:
                content.append({"type": "text", "text": part})

        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=schema,
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RefusedError(getattr(response.stop_details, "explanation", None) or "request declined")
        if response.parsed_output is None:
            raise RuntimeError(f"no structured output (stop_reason={response.stop_reason})")
        return response.parsed_output
