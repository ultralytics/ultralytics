# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import asyncio
import json

import pytest

from ultralytics import LLM, Agent, Gate


def test_agent_sync_execution():
    """Test real synchronous Agent and Gate execution."""
    sequential = Agent(lambda value: value + 1, lambda value: value * 2)
    assert sequential(1) == {"function": [2], "function_2": [4]}
    assert list(sequential(1, stream=True)) == [("function", 2), ("function_2", 4)]

    agent = Agent(
        {
            "detect": lambda value: {"count": value},
            "gate": Gate.when("detect.count", gte=1),
            "describe": lambda value: f"found {value['count']}",
        }
    )
    agent.connect("detect", "gate").connect("gate", "describe")
    assert agent(2)["describe"] == ["found 2"]
    assert agent(0)["describe"] == []

    with pytest.raises(ValueError, match="cycles"):
        agent.connect("describe", "detect")


def test_agent_async_execution():
    """Test real asynchronous callable execution."""

    async def add_one(value):
        """Add one asynchronously."""
        await asyncio.sleep(0)
        return value + 1

    async def double(value):
        """Double a value asynchronously."""
        await asyncio.sleep(0)
        return value * 2

    agent = Agent({"add": add_one, "double": double}).connect("add", "double")
    assert asyncio.run(agent.async_call(1)) == {"add": [2], "double": [4]}


def test_agent_serialization_omits_credentials():
    """Test real Agent serialization without credentials."""
    agent = Agent(
        {
            "llm": LLM(
                "gpt-5.6-luna", api_key="secret", extra_headers={"Authorization": "Bearer header-secret"}, temperature=0
            ),
            "gate": Gate.every(seconds=30),
        }
    ).connect("llm", "gate")

    definition = agent.to_dict()
    assert "secret" not in json.dumps(definition)
    assert Agent.from_dict(definition).to_dict() == definition

    definition["blocks"][0]["config"]["extra_headers"] = {"Authorization": "Bearer secret"}
    with pytest.raises(ValueError, match="credentials"):
        Agent.from_dict(definition)
