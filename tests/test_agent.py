# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ultralytics import LLM, YOLO, Agent, Gate


def test_gate_evaluation():
    """Test condition and cadence Gate composition."""
    gate = Gate.all(Gate.when("detections.count", gte=1), Gate.every(frames=2))

    assert gate({"detections": {"count": 1}}) is None
    assert gate({"detections": {"count": 2}}) == {"detections": {"count": 2}}
    assert gate({"detections": {"count": 0}}) is None
    assert Gate.when("values", gte=1)({"values": np.array([1, 2])}) is None


def test_agent_execution_and_cycle_validation():
    """Test synchronous graph execution and cycle rejection."""
    sequential = Agent(lambda value: value + 1, lambda value: value * 2)
    assert list(sequential.blocks) == ["function", "function_2"]
    assert sequential(1) == {"function": [2], "function_2": [4]}

    agent = Agent(
        {
            "detect": lambda value: {"count": value},
            "gate": Gate.when("detect.count", gte=1),
            "describe": lambda value: f"found {value['count']}",
        }
    )
    agent.connect("detect", "gate").connect("gate", "describe")

    assert agent(2) == {"detect": [{"count": 2}], "gate": [{"count": 2}], "describe": ["found 2"]}
    assert agent(0) == {"detect": [{"count": 0}], "gate": [], "describe": []}
    with pytest.raises(ValueError, match="cycles"):
        agent.connect("describe", "detect")
    with pytest.raises(KeyError, match="Unknown Block"):
        agent.connect("detect.output", "gate")
    with pytest.raises(ValueError, match="at least one"):
        Agent({})


def test_agent_async_execution():
    """Test asynchronous graph execution."""

    class AsyncBlock:
        def __call__(self, value):
            """Return a synchronous value."""
            return value

        async def async_call(self, value):
            """Return an asynchronous value."""
            await asyncio.sleep(0)
            return value + 1

    agent = Agent({"first": AsyncBlock(), "second": AsyncBlock()}).connect("first", "second")
    assert asyncio.run(agent.async_call(1)) == {"first": [2], "second": [3]}


def test_agent_serialization_omits_credentials():
    """Test lossless graph serialization without API keys."""
    agent = Agent(
        {
            "first": LLM("gpt-5.6-luna", api_key="secret", temperature=0),
            "gate": Gate.every(seconds=30, immediate=True),
            "second": LLM("gpt-5.6-luna", base_url="http://localhost:11434/v1"),
        }
    )
    agent.connect("first", "gate").connect("gate", "second")

    definition = agent.to_dict()
    assert "secret" not in json.dumps(definition)
    assert Agent.from_dict(definition).to_dict() == definition


def test_yolo_gate_llm_event_contract(monkeypatch):
    """Test YOLO event standardization, Gate filtering, and LLM vision input."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    def person_summary():
        """Return one serialized person detection."""
        return [{"name": "person", "class": 0, "confidence": 0.9}]

    def empty_summary():
        """Return an empty detection summary."""
        return []

    results = [
        SimpleNamespace(orig_img=image, summary=person_summary),
        SimpleNamespace(orig_img=image, summary=empty_summary),
    ]

    def yolo_call(self, source, **kwargs):
        """Return two image results to exercise event fan-out."""
        return results

    calls = []

    def create(**kwargs):
        """Record a multimodal LLM request."""
        calls.append(kwargs)
        return SimpleNamespace(output_text="A person")

    monkeypatch.setattr(YOLO, "__call__", yolo_call)
    yolo = object.__new__(YOLO)
    llm = LLM(prompt="Describe the people.")
    llm.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    agent = Agent(yolo, Gate.when("yolo.classes.person.count", gte=1), llm)

    output = agent(image)
    assert len(output["yolo"]) == 2
    assert output["llm"] == [{"text": "A person"}]
    assert len(calls) == 1
    content = calls[0]["input"][0]["content"]
    assert "person" in content[0]["text"]
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")

    calls.clear()
    direct = Agent(yolo, llm)(image)
    assert len(direct["yolo"]) == len(direct["llm"]) == len(calls) == 2


def test_llm_sync_and_async_calls():
    """Test LLM request routing for sync and async clients."""
    calls = []

    def create(**kwargs):
        """Record a synchronous request."""
        calls.append(kwargs)
        return "sync"

    async def async_create(**kwargs):
        """Record an asynchronous request."""
        calls.append(kwargs)
        return "async"

    llm = LLM(temperature=0)
    llm.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    llm.async_client = SimpleNamespace(responses=SimpleNamespace(create=async_create))

    assert llm("hello") == "sync"
    assert asyncio.run(llm.async_call("hello", stream=True)) == "async"
    assert calls == [
        {"temperature": 0, "model": "gpt-5.6-luna", "input": "hello"},
        {"temperature": 0, "stream": True, "model": "gpt-5.6-luna", "input": "hello"},
    ]

    chat = LLM(api="chat.completions")
    chat.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert chat("hello") == "sync"
    assert calls[-1] == {
        "model": "gpt-5.6-luna",
        "messages": [{"role": "user", "content": "hello"}],
    }

    assert llm("hello", model="custom") == "sync"
    assert calls[-1]["model"] == "custom"
