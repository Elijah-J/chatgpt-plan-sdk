"""Terminal Responses result and model-catalog projections."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from chatgpt_plan_sdk._errors import ResponsesAPIError

__all__ = ["ModelCatalog", "ResponsesResult"]


def _counter(usage: Any, *path: str) -> int | None:
    # A JSON true/false is not a counter: the exact type must be int.
    node = usage
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if type(node) is int else None


@dataclass(frozen=True, slots=True)
class ResponsesResult:
    """One completed response: the untouched terminal object and its items.

    ``response`` is the terminal ``response`` object as fed. ``output`` holds
    the completed output items, assembled from ``response.output_item.done``
    events ordered by output index, or copied from the terminal response's
    output when no item events arrived. Inputs are deep-copied.
    """

    response: dict[str, Any]
    output: tuple[dict[str, Any], ...]
    event_counts: dict[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "response", copy.deepcopy(self.response))
        object.__setattr__(self, "output", tuple(copy.deepcopy(item) for item in self.output))
        object.__setattr__(self, "event_counts", copy.deepcopy(self.event_counts))

    def _text(self, key: str) -> str | None:
        value = self.response.get(key)
        return value if isinstance(value, str) else None

    @property
    def id(self) -> str | None:
        return self._text("id")

    @property
    def model(self) -> str | None:
        return self._text("model")

    @property
    def status(self) -> str | None:
        return self._text("status")

    @property
    def usage(self) -> dict[str, Any] | None:
        usage = self.response.get("usage")
        return usage if isinstance(usage, dict) else None

    @property
    def input_tokens(self) -> int | None:
        return _counter(self.usage, "input_tokens")

    @property
    def cached_tokens(self) -> int | None:
        return _counter(self.usage, "input_tokens_details", "cached_tokens")

    @property
    def cache_write_tokens(self) -> int | None:
        return _counter(self.usage, "input_tokens_details", "cache_write_tokens")

    @property
    def output_tokens(self) -> int | None:
        return _counter(self.usage, "output_tokens")

    @property
    def reasoning_tokens(self) -> int | None:
        return _counter(self.usage, "output_tokens_details", "reasoning_tokens")

    @property
    def total_tokens(self) -> int | None:
        return _counter(self.usage, "total_tokens")

    def _parts(self, kind: str, field: str) -> list[str]:
        found: list[str] = []
        for item in self.output:
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == kind:
                    value = part.get(field)
                    if isinstance(value, str):
                        found.append(value)
        return found

    @property
    def output_text(self) -> str:
        """Message content parts of type ``output_text``, joined in order."""
        return "".join(self._parts("output_text", "text"))

    @property
    def refusals(self) -> tuple[str, ...]:
        """Message content parts of type ``refusal``, in order."""
        return tuple(self._parts("refusal", "refusal"))

    def output_json(self) -> Any:
        """Decode ``output_text`` as JSON; the SDK validates no schema.

        Raises ``ValueError`` when any refusal part exists or when the text is
        empty or not valid JSON.
        """
        if self.refusals:
            raise ValueError("response contains a refusal")
        return json.loads(self.output_text)

    def to_input_items(self) -> list[dict[str, Any]]:
        """A new list of deep copies of every completed output item, in order.

        Items are returned unchanged (unknown keys, ``phase``,
        ``encrypted_content`` and unknown item types included), ready to be
        appended to the next request's ``input`` array.
        """
        return [copy.deepcopy(item) for item in self.output]


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """The native ``models`` array of a successful ``GET /v1/models``."""

    models: tuple[dict[str, Any], ...]

    @classmethod
    def from_response(cls, response: httpx.Response) -> ModelCatalog:
        """Read a catalog response; a non-2xx status raises ``ResponsesAPIError``."""
        if not response.is_success:
            raise ResponsesAPIError(response)
        body = response.json()
        models = body.get("models") if isinstance(body, Mapping) else None
        if not isinstance(models, list) or not all(isinstance(m, dict) for m in models):
            raise ValueError("model catalog response has no models array of objects")
        return cls(models=tuple(copy.deepcopy(models)))

    def visible_slugs(self) -> tuple[str, ...]:
        """Slugs of models whose ``visibility`` is ``list``, in catalog order."""
        return tuple(
            model["slug"]
            for model in self.models
            if model.get("visibility") == "list" and isinstance(model.get("slug"), str)
        )
