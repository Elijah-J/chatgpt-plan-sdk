"""Offline checks for ``ResponsesResult.output_json`` and ``to_input_items``."""

import json

import pytest

from chatgpt_plan_sdk import ResponsesResult


def message(parts=(), *, refusal=None, **extra):
    content = [{"type": "output_text", "text": part} for part in parts]
    if refusal is not None:
        content.append({"type": "refusal", "refusal": refusal})
    return {"type": "message", "role": "assistant", "content": content, **extra}


def result_of(items, raw_output=()):
    return ResponsesResult(
        response={"id": "r", "status": "completed", "output": list(raw_output)},
        output=tuple(items),
        event_counts={},
    )


def test_output_json_decodes_joined_text_without_validating_a_schema():
    assert result_of([message(['{"value": ', "3}"])]).output_json() == {"value": 3}
    assert result_of([message(['["a", 1, null]'])]).output_json() == ["a", 1, None]
    assert result_of([message(['{"value": "x", "extra": true}'])]).output_json() == {
        "value": "x",
        "extra": True,
    }


@pytest.mark.parametrize(
    "items",
    [
        [message(['{"value": 3'])],
        [message([""])],
        [{"type": "reasoning", "summary": []}],
        [message(['{"value": 3}'], refusal="no")],
    ],
    ids=["truncated", "empty-text", "no-message", "refusal-with-json"],
)
def test_output_json_raises_value_error(items):
    with pytest.raises(ValueError):
        result_of(items).output_json()


def test_to_input_items_copies_every_item_in_order_unchanged():
    items = [
        {"type": "reasoning", "summary": [], "encrypted_content": "enc", "x_new": {"k": [1, None]}},
        message(['{"value": 3}'], phase="final_answer"),
        {"type": "x_future", "payload": [3, 2, 1]},
    ]
    result = result_of(items, raw_output=[{"type": "from-response-output-only"}])
    replay = result.to_input_items()
    assert type(replay) is list
    assert replay == items
    assert [list(item) for item in replay] == [list(item) for item in items]
    replay[0]["encrypted_content"] = "changed"
    replay[1]["content"][0]["text"] = "changed"
    replay[2]["payload"].append(0)
    replay.append({"type": "extra"})
    assert result.to_input_items() == items
    assert json.dumps(list(result.output)) == json.dumps(items)


def test_to_input_items_of_an_empty_output_is_a_new_empty_list():
    result = result_of([])
    first = result.to_input_items()
    assert first == []
    first.append({})
    assert result.to_input_items() == []
