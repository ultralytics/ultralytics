# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import asyncio
import inspect
import json
import operator
import time
from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from functools import partial
from typing import Any, ClassVar


class Gate:
    """Stateful Block that forwards signals matching condition or cadence rules.

    Attributes:
        definition (dict): JSON-compatible Gate configuration.
        state (dict): Runtime cadence state scoped to this Gate instance.
        ports (dict): Agent input and output port definitions.

    Methods:
        every: Create a time-based or frame-based cadence Gate.
        when: Create a conditional Gate for a signal field.
        all: Create a Gate requiring every child Gate to pass.
        any: Create a Gate requiring at least one child Gate to pass.
        __call__: Forward a passing signal or return None.

    Examples:
        >>> from ultralytics import Gate
        >>> gate = Gate.all(Gate.when("person_count", gte=1), Gate.every(frames=10))
        >>> output = gate({"person_count": 2})
    """

    ports: ClassVar = {"inputs": {"event": "event"}, "outputs": {"event": "event"}}

    OPS: ClassVar = {
        "eq": operator.eq,
        "ne": operator.ne,
        "gt": operator.gt,
        "gte": operator.ge,
        "lt": operator.lt,
        "lte": operator.le,
        "in": lambda current, expected: current in expected,
        "contains": lambda current, expected: expected in current,
    }

    def __init__(self, definition: Mapping[str, Any]) -> None:
        """Initialize a Gate from its JSON-compatible definition."""
        self.definition = deepcopy(dict(definition))
        self._validate(self.definition)
        self.state: dict[str, Any] = {}

    @classmethod
    def every(cls, *, seconds: float | None = None, frames: int | None = None, immediate: bool = False) -> Gate:
        """Forward every N seconds or frames."""
        if (seconds is None) == (frames is None):
            raise ValueError("Specify exactly one of 'seconds' or 'frames'.")
        unit, value = ("seconds", seconds) if seconds is not None else ("frames", frames)
        return cls({"kind": "every", "unit": unit, "value": value, "immediate": immediate})

    @classmethod
    def when(cls, path: str, **condition: Any) -> Gate:
        """Forward matching signals, for example `Gate.when("count", gte=1)`."""
        if len(condition) != 1:
            raise ValueError(f"Specify exactly one condition operator from {(*tuple(cls.OPS), 'exists')}.")
        op, value = next(iter(condition.items()))
        return cls({"kind": "condition", "path": path, "op": op, "value": value})

    @classmethod
    def all(cls, *gates: Gate) -> Gate:
        """Forward signals matching every Gate."""
        return cls._combine("all", gates)

    @classmethod
    def any(cls, *gates: Gate) -> Gate:
        """Forward signals matching at least one Gate."""
        return cls._combine("any", gates)

    def __call__(self, signal: Any) -> Any | None:
        """Return the input signal when it passes, otherwise return None."""
        return signal if self._evaluate(self.definition, signal, self.state) else None

    def _agent_run(self, event: dict[str, Any], name: str, **kwargs: Any) -> dict[str, Any] | None:
        """Filter an Agent event without removing its accumulated context."""
        return self(event)

    def to_dict(self) -> dict[str, Any]:
        """Return this Gate's JSON-compatible configuration."""
        return deepcopy(self.definition)

    @classmethod
    def _combine(cls, kind: str, gates: tuple[Gate, ...]) -> Gate:
        """Create a composite Gate."""
        if not gates or not all(isinstance(gate, cls) for gate in gates):
            raise ValueError(f"Gate.{kind}() requires one or more Gate instances.")
        return cls({"kind": kind, "gates": [gate.to_dict() for gate in gates]})

    @classmethod
    def _validate(cls, definition: dict[str, Any]) -> None:
        """Validate a serialized Gate definition."""
        kind = definition.get("kind")
        if kind == "every":
            unit, value = definition.get("unit"), definition.get("value")
            if (
                unit not in {"seconds", "frames"}
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
                or (unit == "frames" and not isinstance(value, int))
            ):
                raise ValueError("Every Gates require a positive seconds value or integer frame count.")
            if not isinstance(definition.get("immediate", False), bool):
                raise ValueError("Gate 'immediate' must be a boolean.")
            return
        if kind == "condition":
            if not isinstance(definition.get("path"), str) or not definition["path"]:
                raise ValueError("Condition Gates require a field path.")
            if definition.get("op") not in {*cls.OPS, "exists"}:
                raise ValueError(f"Unsupported condition operator {definition.get('op')!r}.")
            return
        if kind in {"all", "any"}:
            gates = definition.get("gates")
            if not isinstance(gates, list) or not gates:
                raise ValueError(f"Gate.{kind}() requires one or more Gates.")
            for gate in gates:
                if not isinstance(gate, dict):
                    raise TypeError("Child Gates must be objects.")
                cls._validate(gate)
            return
        raise ValueError(f"Unsupported Gate kind {kind!r}.")

    @classmethod
    def _evaluate(cls, definition: dict[str, Any], signal: Any, state: dict[str, Any]) -> bool:
        """Evaluate a validated Gate definition."""
        kind = definition["kind"]
        if kind == "every":
            immediate = definition.get("immediate", False)
            if definition["unit"] == "frames":
                state["count"] = state.get("count", 0) + 1
                return (immediate and state["count"] == 1) or state["count"] % definition["value"] == 0
            now = time.monotonic()
            if "last" not in state:
                state["last"] = now
                return immediate
            if now - state["last"] >= definition["value"]:
                state["last"] = now
                return True
            return False
        if kind == "condition":
            value = cls._resolve(signal, definition["path"])
            if definition["op"] == "exists":
                return (value is not _MISSING) == bool(definition.get("value"))
            if value is _MISSING:
                return False
            try:
                return bool(cls.OPS[definition["op"]](value, definition.get("value")))
            except (RuntimeError, TypeError, ValueError):
                return False
        results = [
            cls._evaluate(gate, signal, state.setdefault(str(i), {})) for i, gate in enumerate(definition["gates"])
        ]
        return all(results) if kind == "all" else any(results)

    @staticmethod
    def _resolve(signal: Any, path: str) -> Any:
        """Resolve a dotted path from mappings, objects, or sequences."""
        value = signal
        for part in path.split("."):
            if isinstance(value, Mapping):
                value = value.get(part, _MISSING)
            elif isinstance(value, (list, tuple)) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                value = getattr(value, part, _MISSING)
            if value is _MISSING:
                break
        return value


class Agent:
    """Executable directed graph of callable Blocks.

    Attributes:
        blocks (dict): Callable Blocks keyed by stable names.
        connections (list): Passive source-target Block pairs.

    Methods:
        connect: Connect one named Block output to another Block input.
        __call__: Execute the graph synchronously.
        async_call: Execute graph frontiers asynchronously.
        to_dict: Serialize the graph without credentials.
        from_dict: Create an Agent from a serialized graph.

    Examples:
        Create a sequential Agent by passing Blocks in execution order:
        >>> from ultralytics import Agent, Gate, LLM, YOLO
        >>> agent = Agent(
        ...     YOLO("yolo26n.pt"),
        ...     Gate.when("yolo.classes.person.count", gte=1),
        ...     LLM("gpt-5.6-luna", prompt="Describe the people."),
        ... )

        Name and connect Blocks explicitly for branching or stable identifiers:
        >>> agent = Agent({"detect": YOLO("yolo26n.pt"), "describe": LLM("gpt-5.6-luna", prompt="Describe the image.")})
        >>> agent.connect("detect", "describe")
    """

    SCHEMA_VERSION = 1

    def __init__(self, *blocks: Callable[..., Any] | Mapping[str, Callable[..., Any]]) -> None:
        """Initialize an Agent from sequential Blocks or one mapping of named Blocks."""
        sequential = not (len(blocks) == 1 and isinstance(blocks[0], Mapping))
        if sequential:
            named_blocks = {}
            for block in blocks:
                name = type(block).__name__.lower()
                suffix = 2
                while name in named_blocks:
                    name = f"{type(block).__name__.lower()}_{suffix}"
                    suffix += 1
                named_blocks[name] = block
        else:
            named_blocks = dict(blocks[0])
        if not named_blocks:
            raise ValueError("Agent requires at least one Block.")
        if any(
            not isinstance(name, str) or not name or name in {"data", "source"} or "." in name for name in named_blocks
        ):
            raise ValueError("Block names must be non-empty strings without dots and cannot be 'data' or 'source'.")
        if any(not callable(block) for block in named_blocks.values()):
            raise TypeError("Every Agent Block must be callable.")
        self.blocks = named_blocks
        self.connections: list[tuple[str, str]] = []
        if sequential:
            names = list(self.blocks)
            self.connections = list(zip(names, names[1:]))

    def connect(self, source: str, target: str) -> Agent:
        """Connect one Block output to another Block input."""
        if source not in self.blocks or target not in self.blocks:
            raise KeyError(f"Unknown Block in connection {source!r} -> {target!r}.")
        self.connections.append((source, target))
        if self._has_cycle():
            self.connections.pop()
            raise ValueError("Agent graphs cannot contain cycles.")
        return self

    def __call__(self, source: Any = None, **kwargs: Any) -> dict[str, list[Any]]:
        """Execute the graph synchronously and group emissions by Block name."""
        outputs: dict[str, list[Any]] = {name: [] for name in self.blocks}
        queue = deque((name, {"source": source, "data": source}, kwargs) for name in self._roots())
        while queue:
            name, event, call_kwargs = queue.popleft()
            for emitted in self._invoke(name, event, call_kwargs):
                outputs[name].append(emitted.get(name, emitted["data"]))
                self._propagate(name, emitted, queue)
        return outputs

    async def async_call(self, source: Any = None, **kwargs: Any) -> dict[str, list[Any]]:
        """Execute each graph frontier asynchronously."""
        outputs: dict[str, list[Any]] = {name: [] for name in self.blocks}
        frontier = [(name, {"source": source, "data": source}, kwargs) for name in self._roots()]
        locks = {name: asyncio.Lock() for name in self.blocks}
        while frontier:
            current, frontier = frontier, []
            results = await asyncio.gather(
                *(self._invoke_async(name, value, call_kwargs, locks[name]) for name, value, call_kwargs in current)
            )
            queue = deque()
            for (name, _, _), emitted_events in zip(current, results):
                for emitted in emitted_events:
                    outputs[name].append(emitted.get(name, emitted["data"]))
                    self._propagate(name, emitted, queue)
            frontier.extend(queue)
        return outputs

    def to_dict(self) -> dict[str, Any]:
        """Serialize this Agent to a JSON-compatible definition without credentials."""
        definition = {
            "schemaVersion": self.SCHEMA_VERSION,
            "blocks": [{"id": name, **self._serialize_block(block)} for name, block in self.blocks.items()],
            "connections": [{"from": source, "to": target} for source, target in self.connections],
        }
        json.dumps(definition)
        return definition

    @classmethod
    def from_dict(cls, definition: Mapping[str, Any]) -> Agent:
        """Create an Agent from a serialized definition."""
        if definition.get("schemaVersion") != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported Agent schema version {definition.get('schemaVersion')!r}.")
        blocks = {}
        for item in definition.get("blocks", []):
            block_id, block_type, config = item.get("id"), item.get("type"), item.get("config", {})
            if block_type == "YOLO":
                from ultralytics.models import YOLO

                blocks[block_id] = YOLO(**config)
            elif block_type == "LLM":
                from ultralytics.models import LLM

                blocks[block_id] = LLM(**config)
            elif block_type == "Gate":
                blocks[block_id] = Gate(config)
            else:
                raise ValueError(f"Unsupported Block type {block_type!r}.")
        agent = cls(blocks)
        for connection in definition.get("connections", []):
            agent.connect(connection["from"], connection["to"])
        return agent

    def _roots(self) -> list[str]:
        """Return Blocks with no incoming connections."""
        targets = {target for _, target in self.connections}
        return [name for name in self.blocks if name not in targets]

    def _propagate(self, source: str, event: dict[str, Any], queue: deque) -> None:
        """Queue connected downstream Blocks."""
        for connection_source, target in self.connections:
            if connection_source == source:
                queue.append((target, {**event}, {}))

    def _invoke(self, name: str, event: dict[str, Any], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Invoke one Block and normalize its emissions to Agent events."""
        block = self.blocks[name]
        if callable(agent_run := getattr(block, "_agent_run", None)):
            return self._events(agent_run(event, name, **kwargs))
        output = block(event["data"], **kwargs)
        return [] if output is None else [{**event, "data": output, name: output}]

    async def _invoke_async(
        self, name: str, event: dict[str, Any], kwargs: dict[str, Any], lock: asyncio.Lock
    ) -> list[dict[str, Any]]:
        """Invoke an async-capable Block without concurrently re-entering the same instance."""
        block = self.blocks[name]
        async with lock:
            if callable(agent_async_run := getattr(block, "_agent_async_run", None)):
                return self._events(await agent_async_run(event, name, **kwargs))
            if callable(agent_run := getattr(block, "_agent_run", None)):
                loop = asyncio.get_running_loop()
                return self._events(await loop.run_in_executor(None, partial(agent_run, event, name, **kwargs)))
            if callable(async_call := getattr(block, "async_call", None)):
                output = await async_call(event["data"], **kwargs)
                return [] if output is None else [{**event, "data": output, name: output}]
            if inspect.iscoroutinefunction(block) or inspect.iscoroutinefunction(type(block).__call__):
                output = await block(event["data"], **kwargs)
            else:
                output = await asyncio.get_running_loop().run_in_executor(None, partial(block, event["data"], **kwargs))
            return [] if output is None else [{**event, "data": output, name: output}]

    @staticmethod
    def _events(output: Any) -> list[dict[str, Any]]:
        """Normalize one or many internal Block emissions to a list."""
        if output is None:
            return []
        return output if isinstance(output, list) else [output]

    def _has_cycle(self) -> bool:
        """Return whether the graph contains a directed cycle."""
        graph = {name: [] for name in self.blocks}
        for source, target in self.connections:
            graph[source].append(target)
        visiting, visited = set(), set()

        def visit(name: str) -> bool:
            """Visit one Block while detecting back edges."""
            if name in visiting:
                return True
            if name in visited:
                return False
            visiting.add(name)
            cyclic = any(visit(target) for target in graph[name])
            visiting.remove(name)
            visited.add(name)
            return cyclic

        return any(visit(name) for name in graph)

    @staticmethod
    def _serialize_block(block: Callable[..., Any]) -> dict[str, Any]:
        """Serialize a supported built-in Block without credentials."""
        from ultralytics.models import LLM, YOLO

        if isinstance(block, LLM):
            config = {"model": block.model, "api": block.api, **block.overrides}
            if block.base_url is not None:
                config["base_url"] = block.base_url
            if block.prompt is not None:
                config["prompt"] = block.prompt
            return {"type": "LLM", "config": config, "ports": deepcopy(block.ports)}
        if isinstance(block, YOLO):
            return {
                "type": "YOLO",
                "config": {"model": str(block.model_name), "task": block.task},
                "ports": deepcopy(block.ports),
            }
        if isinstance(block, Gate):
            return {"type": "Gate", "config": block.to_dict(), "ports": deepcopy(block.ports)}
        raise TypeError(f"Block type {type(block).__name__!r} is executable but not serializable.")


_MISSING = object()
