# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import asyncio
import json
from types import SimpleNamespace

import pytest

from ultralytics import LLM, Agent, Gate


def test_gate_evaluation():
    gate = Gate.all(Gate.when("detections.count", gte=1), Gate.every(frames=2))

    assert gate({"detections": {"count": 1}}) is None
    assert gate({"detections": {"count": 2}}) == {"detections": {"count": 2}}
    assert gate({"detections": {"count": 0}}) is None


def test_agent_execution_and_cycle_validation():
    agent = Agent(
        {
            "detect": lambda value: {"count": value},
            "gate": Gate.when("count", gte=1),
            "describe": lambda value: f"found {value['count']}",
        }
    )
    agent.connect("detect", "gate").connect("gate", "describe")

    assert agent(2) == {"detect": [{"count": 2}], "gate": [{"count": 2}], "describe": ["found 2"]}
    assert agent(0) == {"detect": [{"count": 0}], "gate": [], "describe": []}
    with pytest.raises(ValueError, match="cycles"):
        agent.connect("describe", "detect")


def test_agent_async_execution():
    class AsyncBlock:
        def __call__(self, value):
            return value

        async def async_call(self, value):
            await asyncio.sleep(0)
            return value + 1

    agent = Agent({"first": AsyncBlock(), "second": AsyncBlock()}).connect("first", "second")
    assert asyncio.run(agent.async_call(1)) == {"first": [2], "second": [3]}


def test_agent_serialization_omits_credentials():
    agent = Agent(
        {
            "first": LLM("gpt-5.6-luna", api_key="secret", temperature=0),
            "gate": Gate.every(seconds=30, immediate=True),
            "second": LLM("gpt-5.6-luna", base_url="http://localhost:11434/v1"),
        }
    )
    agent.connect("first.response", "gate.input").connect("gate.output", "second.input")

    definition = agent.to_dict()
    assert "secret" not in json.dumps(definition)
    assert Agent.from_dict(definition).to_dict() == definition


def test_llm_sync_and_async_calls():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return "sync"

    async def async_create(**kwargs):
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
