# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import asyncio
import base64
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from ultralytics import LLM, YOLO, Agent, Gate


def test_gate_evaluation():
    """Test condition and cadence Gate composition."""
    gate = Gate.all(Gate.when("detections.count", gte=1), Gate.every(frames=2))

    assert gate({"detections": {"count": 1}}) is None
    assert gate({"detections": {"count": 2}}) == {"detections": {"count": 2}}
    assert gate({"detections": {"count": 0}}) is None
    assert Gate.when("values", gte=1)({"values": np.array([1, 2])}) is None
    immediate = Gate.every(frames=3, immediate=True)
    assert [immediate(i) is not None for i in range(7)] == [True, False, False, True, False, False, True]


def test_agent_execution_and_cycle_validation():
    """Test synchronous graph execution and cycle rejection."""
    sequential = Agent(lambda value: value + 1, lambda value: value * 2)
    assert list(sequential.blocks) == ["function", "function_2"]
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

    assert agent(2) == {"detect": [{"count": 2}], "gate": [{"passed": True}], "describe": ["found 2"]}
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

    async def stream():
        """Collect asynchronous graph emissions."""
        return [event async for event in agent.async_call(1, stream=True)]

    assert asyncio.run(stream()) == [("first", 2), ("second", 3)]

    with pytest.raises(ValueError, match="distinct instance"):
        Agent({"first": agent.blocks["first"], "second": agent.blocks["first"]})

    class StreamBlock:
        def __init__(self):
            """Initialize stream activity state."""
            self.active = False

        def __call__(self, value):
            """Return a synchronous value."""
            return value

        def _agent_run(self, event, name, **kwargs):
            """Emit one event while rejecting concurrent stream entry."""
            assert not self.active
            self.active = True
            try:
                yield {**event, name: event["data"]}
            finally:
                self.active = False

    fan_in = Agent({"a": lambda value: value, "b": lambda value: value, "stream": StreamBlock()})
    fan_in.connect("a", "stream").connect("b", "stream")
    assert asyncio.run(fan_in.async_call(1))["stream"] == [1, 1]

    async def async_block(value):
        """Return an asynchronous function result."""
        return value

    with pytest.raises(TypeError, match=r"Agent.async_call"):
        Agent(async_block)(1)


def test_agent_serialization_omits_credentials():
    """Test lossless graph serialization without API keys."""
    agent = Agent(
        {
            "first": LLM(
                "gpt-5.6-luna", api_key="secret", extra_headers={"Authorization": "Bearer header-secret"}, temperature=0
            ),
            "gate": Gate.every(seconds=30, immediate=True),
            "second": LLM("gpt-5.6-luna", base_url="http://localhost:11434/v1"),
        }
    )
    agent.connect("first", "gate").connect("gate", "second")

    definition = agent.to_dict()
    assert "secret" not in json.dumps(definition)
    assert Agent.from_dict(definition).to_dict() == definition
    definition["blocks"][0]["config"]["extra_headers"] = {"Authorization": "Bearer secret"}
    with pytest.raises(ValueError, match="credentials"):
        Agent.from_dict(definition)


def test_yolo_gate_llm_event_contract(monkeypatch):
    """Test YOLO event standardization, Gate filtering, and LLM vision input."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    def person_summary():
        """Return one serialized person detection."""
        return [
            {
                "name": "person",
                "class": 0,
                "confidence": 0.9,
                "box": {"x1": 1, "y1": 1, "x2": 2, "y2": 2},
                "segments": {"x": [1, 2], "y": [3, 4]},
                "keypoints": {"x": [1], "y": [2]},
            }
        ]

    def empty_summary():
        """Return an empty detection summary."""
        return []

    def classification_summary():
        """Return ranked classification predictions rather than instance counts."""
        return [{"name": "ambulance", "class": 1, "confidence": 0.002}]

    results = [
        SimpleNamespace(orig_img=image, probs=None, summary=person_summary),
        SimpleNamespace(orig_img=image, probs=None, summary=empty_summary),
        SimpleNamespace(orig_img=image, probs=object(), summary=classification_summary),
    ]

    consumed = []

    def yolo_call(self, source, **kwargs):
        """Return image results to exercise event fan-out."""
        assert isinstance(source, np.ndarray)
        for result in results:
            consumed.append(result)
            yield result

    calls = []
    observed = []

    def create(**kwargs):
        """Record a multimodal LLM request."""
        calls.append(kwargs)
        observed.append(len(consumed))
        return SimpleNamespace(output_text="A person")

    async def async_create(**kwargs):
        """Record an asynchronous multimodal LLM request."""
        calls.append(kwargs)
        observed.append(len(consumed))
        await asyncio.sleep(0)
        return SimpleNamespace(output_text="A person")

    monkeypatch.setattr(YOLO, "__call__", yolo_call)
    yolo = object.__new__(YOLO)
    llm = LLM(prompt="Describe the people.")
    llm.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    llm.async_client = SimpleNamespace(responses=SimpleNamespace(create=async_create))
    agent = Agent(yolo, Gate.when("yolo.counts.person", gte=1), llm)

    stream = Agent(yolo, llm)(image, stream=True)
    assert next(stream)[0] == "yolo"
    assert len(consumed) == 1 and not calls
    del stream
    consumed.clear()

    output = agent(image)
    assert len(output["yolo"]) == 3
    assert output["yolo"][0] == {
        "results": [
            {
                "name": "person",
                "class": 0,
                "confidence": 0.9,
                "box": {"x1": 1, "y1": 1, "x2": 2, "y2": 2},
            }
        ],
        "counts": {"person": 1},
    }
    assert output["yolo"][2]["counts"] == {}
    assert output["llm"] == [{"text": "A person"}]
    assert output["gate"] == [{"passed": True}]
    assert len(calls) == 1
    assert observed == [1]
    content = calls[0]["input"][0]["content"]
    assert "person" in content[0]["text"]
    assert '"gate"' not in content[0]["text"]
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")

    calls.clear()
    consumed.clear()
    observed.clear()
    direct = Agent(yolo, llm)(image)
    assert len(direct["yolo"]) == len(direct["llm"]) == len(calls) == 3
    assert observed == [1, 2, 3]

    cascade = Agent(yolo, object.__new__(YOLO))(image)
    assert len(cascade["yolo"]) == 3
    assert len(cascade["yolo_2"]) == 9

    calls.clear()
    consumed.clear()
    observed.clear()
    assert asyncio.run(agent.async_call(image)) == output
    assert observed == [1]


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

    with pytest.raises(ValueError, match="Unable to read Agent image"):
        LLM._image_url("missing.jpg")

    encoded = LLM._image_url(Image.new("RGB", (4, 4), (255, 0, 0)))
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded.split(",", 1)[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded[0, 0, 2] > decoded[0, 0, 0]

    prompt = LLM(prompt="Write a summary.")
    prompt.client = llm.client
    assert prompt() == "sync"
    assert calls[-1]["input"] == "Write a summary."
    assert Agent(prompt)()["llm"] == [{"text": "sync"}]
    assert calls[-1]["input"] == "Write a summary."
    assert "array" in prompt._agent_input({"source": None, "data": None, "array": np.ones(2)})

    assert prompt("details") == "sync"
    assert calls[-1]["input"] == "Write a summary.\n\ndetails"
    assert llm("Summarize report.png") == "sync"
    assert calls[-1]["input"] == "Summarize report.png"
    assert llm("https://en.wikipedia.org/wiki/YOLO") == "sync"
    assert calls[-1]["input"] == "https://en.wikipedia.org/wiki/YOLO"
    long_prompt = "x" * 5000
    assert llm(long_prompt) == "sync"
    assert calls[-1]["input"] == long_prompt
    assert llm(Image.new("RGB", (4, 4))) == "sync"
    assert calls[-1]["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
