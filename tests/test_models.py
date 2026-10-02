"""Offline checks for the request, result, catalog and preview-checker models."""

import copy

import httpx
import pytest

from chatgpt_plan_sdk import (
    ModelCatalog,
    ResponsesAPIError,
    ResponsesRequest,
    ResponsesResult,
    siwc_preview_violations,
)


def test_request_body_keeps_only_set_fields_and_native_extras():
    request = ResponsesRequest(
        model="m", input=[{"role": "user", "content": "hi"}], service_tier=None, x_new={"k": [1]}
    )
    assert request.to_body() == {
        "model": "m",
        "input": [{"role": "user", "content": "hi"}],
        "service_tier": None,
        "x_new": {"k": [1]},
    }


@pytest.mark.parametrize("bad", [{"model": 1}, {"input": "text"}, {"input": [3]}])
def test_request_is_strict(bad):
    fields = {"model": "m", "input": []}
    fields.update(bad)
    with pytest.raises(ValueError):
        ResponsesRequest(**fields)


def test_result_projects_paths_and_rejects_bool_counters():
    response = {
        "id": "r1",
        "model": "m",
        "status": "completed",
        "usage": {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": True},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 6,
        },
    }
    output = ({"type": "message", "content": [
        {"type": "output_text", "text": "a"},
        {"type": "refusal", "refusal": "no"},
        {"type": "output_text", "text": "b"},
    ]},)
    result = ResponsesResult(response=response, output=output, event_counts={})
    assert (result.input_tokens, result.cached_tokens, result.cache_write_tokens) == (4, None, None)
    assert (result.output_tokens, result.reasoning_tokens, result.total_tokens) == (2, 1, 6)
    assert result.output_text == "ab" and result.refusals == ("no",)
    response["usage"]["input_tokens"] = 99
    assert result.input_tokens == 4


def test_catalog_lists_visible_slugs_in_order_and_refuses_errors():
    body = {"models": [
        {"slug": "a", "visibility": "list"},
        {"slug": "b", "visibility": "hide"},
        {"slug": "c", "visibility": "list"},
    ]}
    request = httpx.Request("GET", "https://api.openai.com/v1/models")
    catalog = ModelCatalog.from_response(httpx.Response(200, json=body, request=request))
    assert catalog.visible_slugs() == ("a", "c")
    with pytest.raises(ResponsesAPIError) as caught:
        ModelCatalog.from_response(httpx.Response(429, json={"error": {"code": "x"}}, request=request))
    assert "429" in str(caught.value) and "x" not in str(caught.value)


CLEAN = {
    "model": "gpt-6-astra",
    "input": [{"role": "user", "content": "hi"}],
    "instructions": "Be brief.",
    "store": False,
    "reasoning": {"effort": "low"},
}


@pytest.mark.parametrize(
    "change",
    [
        {"store": True},
        {"input": "hi"},
        {"input": [{"role": "system", "content": "x"}]},
        {"previous_response_id": "r"},
        {"temperature": 0.1},
        {"tools": [{"type": "function", "name": "f"}]},
        {"tools": [{"type": "mcp"}]},
        {"reasoning": {"effort": "none"}},
    ],
)
def test_checker_reports_one_violation_per_restriction(change):
    body = {**copy.deepcopy(CLEAN), **change}
    assert len(siwc_preview_violations(body)) == 1


def test_checker_accepts_a_clean_body():
    assert siwc_preview_violations(CLEAN) == ()
