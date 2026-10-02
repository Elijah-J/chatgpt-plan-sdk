"""Request body for ``POST /v1/responses``."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["ResponsesRequest"]


class ResponsesRequest(BaseModel):
    """Data and serialization for one native Responses request.

    ``model`` and ``input`` are validated; every other native field passes
    through unchanged, including explicit JSON null. ``to_body()`` omits fields
    the caller did not supply. A supplied ``stream`` field is refused by
    ``ResponsesClient.stream``, which owns it.
    """

    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)

    model: str
    input: list[dict[str, Any]]

    def to_body(self) -> dict[str, Any]:
        """Return the JSON body of exactly the fields that were set."""
        return self.model_dump(exclude_unset=True)
