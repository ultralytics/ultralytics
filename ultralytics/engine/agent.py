# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import asyncio
import inspect
import json
import operator
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from functools import partial
from typing import Any, ClassVar

_CREDENTIAL_KEYS = {"api_key", "default_headers", "extra_headers", "extra_query"}


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

    ports: ClassVar = {"inputs": {"event": "event"}, "outputs": {"passed": "bool"}}

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
        return {**event, name: {"passed": True}} if self(event) is not None else None

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
                offset = state["count"] - 1 if immediate else state["count"]
                return offset % definition["value"] == 0
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


class Dataset:
    """Dataset Source Block that emits one train, validation, or test split.

    Attributes:
        source (str): Dataset directory, YAML, NDJSON, URL, or ``ul://`` URI.
        split (str): Dataset split emitted to downstream Blocks.
        ports (dict): Agent input and output port definitions.
    """

    ports: ClassVar = {"inputs": {}, "outputs": {"source": "image"}}

    def __init__(self, source: str, split: str = "val") -> None:
        """Initialize a Dataset Block from a directory, YAML, NDJSON, URL, or ``ul://`` URI."""
        if not isinstance(source, str) or not source:
            raise ValueError("Dataset requires a source.")
        if split not in {"train", "val", "test"}:
            raise ValueError("Dataset split must be 'train', 'val', or 'test'.")
        self.source, self.split = source, split

    def __call__(self, source: Any = None) -> Any:
        """Resolve and return the configured dataset split."""
        from pathlib import Path

        from ultralytics.data.utils import check_det_dataset, convert_ndjson_to_yolo_if_needed

        resolved = convert_ndjson_to_yolo_if_needed(self.source)
        if Path(resolved).is_dir():
            split = Path(resolved) / self.split
            if not split.exists():
                raise ValueError(f"Dataset {self.source!r} has no {self.split!r} split.")
            return str(split)
        data = check_det_dataset(str(resolved), split=self.split)
        if not (split := data.get(self.split)):
            raise ValueError(f"Dataset {self.source!r} has no {self.split!r} split.")
        return split

    def _agent_run(self, event: dict[str, Any], name: str, **kwargs: Any) -> dict[str, Any]:
        """Replace the Agent source with the configured dataset split."""
        source = self()
        return {**event, "source": source, "data": source, name: {"source": source}}


class Image:
    """Image Source Block for a local path, URL, array, or PIL image.

    Attributes:
        source (Any): Image source accepted by YOLO prediction.
        ports (dict): Agent input and output port definitions.
    """

    ports: ClassVar = {"inputs": {}, "outputs": {"source": "image"}}

    def __init__(self, source: Any, api_key: str | None = None) -> None:
        """Initialize an Image Block from any source accepted by YOLO prediction."""
        import os

        if source is None:
            raise ValueError("Image requires a source.")
        self.source, self._api_key = source, api_key or os.getenv("IMAGE_API_KEY")

    def __call__(self, source: Any = None) -> Any:
        """Return the configured image source."""
        if isinstance(self.source, str) and self.source.startswith(("http://", "https://")):
            from io import BytesIO

            import requests
            from PIL import Image as PILImage

            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
            response = requests.get(self.source, headers=headers, timeout=(10, 90))
            response.raise_for_status()
            image = PILImage.open(BytesIO(response.content))
            image.load()
            return image
        return self.source

    def _agent_run(self, event: dict[str, Any], name: str, **kwargs: Any) -> dict[str, Any]:
        """Replace the Agent source with the configured image."""
        source = self()
        return {**event, "source": source, "data": source, name: {"source": source}}


class Export:
    """Model export Block that emits a local exported artifact path.

    Attributes:
        model (str): Model path, URL, name, or ``ul://`` URI.
        format (str): Ultralytics export format.
        overrides (dict): Additional model export arguments.
        ports (dict): Agent input and output port definitions.
    """

    ports: ClassVar = {"inputs": {}, "outputs": {"artifact": "model"}}

    def __init__(self, model: str, format: str = "onnx", **kwargs: Any) -> None:
        """Initialize an Export Block from a model and export arguments."""
        if not isinstance(model, str) or not model:
            raise ValueError("Export requires a model.")
        if not isinstance(format, str) or not format:
            raise ValueError("Export requires a format.")
        self.model, self.format, self.overrides = model, format, kwargs

    def __call__(self, source: Any = None) -> Any:
        """Export the configured model and return its artifact path."""
        from ultralytics.models import YOLO

        return YOLO(self.model).export(format=self.format, **self.overrides)


class Deployment:
    """Remote inference Block for any HTTP deployment endpoint.

    Attributes:
        url (str): Base URL of an endpoint exposing ``POST /predict``.
        ports (dict): Agent input and output port definitions.
    """

    ports: ClassVar = {"inputs": {"source": "image"}, "outputs": {"results": "json"}}

    def __init__(self, url: str, api_key: str | None = None) -> None:
        """Initialize a Deployment Block from an HTTP endpoint and optional bearer token."""
        import os

        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError("Deployment requires an HTTP URL.")
        self.url, self._api_key = url.rstrip("/"), api_key or os.getenv("DEPLOYMENT_API_KEY")

    def __call__(self, source: Any, **kwargs: Any) -> Any:
        """Send one image to the configured deployment endpoint."""
        from io import BytesIO
        from pathlib import Path

        import requests
        from PIL import Image as PILImage

        if isinstance(source, str) and source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=(10, 90))
            response.raise_for_status()
            file, filename = BytesIO(response.content), Path(source).name or "image.jpg"
        elif isinstance(source, (str, Path)):
            path = Path(source)
            file, filename = BytesIO(path.read_bytes()), path.name
        else:
            image = source if isinstance(source, PILImage.Image) else PILImage.fromarray(source)
            file = BytesIO()
            image.save(file, format="JPEG")
            file.seek(0)
            filename = "image.jpg"
        try:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
            response = requests.post(
                f"{self.url}/predict",
                headers=headers,
                files={"file": (filename, file)},
                data={key: value for key, value in kwargs.items() if key in {"conf", "iou", "imgsz"}},
                timeout=(10, 90),
            )
            response.raise_for_status()
            return response.json()
        finally:
            file.close()


class Agent:
    """Executable directed graph of callable Blocks.

    Attributes:
        blocks (dict): Callable Blocks keyed by stable names.
        connections (list): Passive source-target Block pairs.

    Methods:
        connect: Connect one named Block output to another Block input.
        __call__: Execute the graph synchronously, optionally as an event stream.
        async_call: Execute the graph asynchronously, optionally as an event stream.
        to_dict: Serialize the graph without credentials.
        from_dict: Create an Agent from a serialized graph.

    Examples:
        Create a sequential Agent by passing Blocks in execution order:
        >>> from ultralytics import Agent, Gate, LLM, YOLO
        >>> agent = Agent(
        ...     YOLO("yolo26n.pt"),
        ...     Gate.when("yolo.counts.person", gte=1),
        ...     LLM("gpt-5.6-luna", prompt="Describe the people."),
        ... )

        Name and connect Blocks explicitly for branching or stable identifiers:
        >>> agent = Agent({"detect": YOLO("yolo26n.pt"), "describe": LLM("gpt-5.6-luna", prompt="Describe the image.")})
        >>> agent.connect("detect", "describe")

        Stream live graph emissions as `(block_name, payload)` pairs:
        >>> for name, payload in agent(0, stream=True):
        ...     print(name, payload)
    """

    SCHEMA_VERSION = 1

    def __init__(self, *blocks: Callable[..., Any] | Mapping[str, Callable[..., Any]]) -> None:
        """Initialize an Agent from sequential Blocks or one mapping of named Blocks."""
        from ultralytics.engine.model import Model

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
        states = {id(block.__dict__) if isinstance(block, Model) else id(block) for block in named_blocks.values()}
        if len(states) != len(named_blocks):
            raise ValueError("Each Agent Block must be a distinct instance.")
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

    def __call__(
        self, source: Any = None, stream: bool = False, **kwargs: Any
    ) -> dict[str, list[Any]] | Iterator[tuple[str, Any]]:
        """Execute synchronously, returning grouped outputs or a stream of `(Block, payload)` pairs."""
        events = self._stream(source, kwargs)
        if stream:
            return events
        outputs: dict[str, list[Any]] = {name: [] for name in self.blocks}
        for name, payload in events:
            outputs[name].append(payload)
        return outputs

    def async_call(
        self, source: Any = None, stream: bool = False, **kwargs: Any
    ) -> Awaitable[dict[str, list[Any]]] | AsyncIterator[tuple[str, Any]]:
        """Execute asynchronously, returning an awaitable result or an async event stream."""
        return self._stream_async(source, kwargs) if stream else self._collect_async(source, kwargs)

    async def _collect_async(self, source: Any, kwargs: dict[str, Any]) -> dict[str, list[Any]]:
        """Execute the graph asynchronously and group emissions by Block name."""
        outputs: dict[str, list[Any]] = {name: [] for name in self.blocks}
        locks: dict[int, asyncio.Lock] = {}
        await asyncio.gather(
            *(
                self._execute_async(name, {"source": source, "data": source}, kwargs, outputs, locks)
                for name in self._roots()
            )
        )
        return outputs

    def _stream(self, source: Any, kwargs: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        """Yield synchronous graph emissions as `(Block, payload)` pairs."""
        for name in self._roots():
            yield from self._execute(name, {"source": source, "data": source}, kwargs)

    async def _stream_async(self, source: Any, kwargs: dict[str, Any]) -> AsyncIterator[tuple[str, Any]]:
        """Yield asynchronous graph emissions as `(Block, payload)` pairs."""
        queue: asyncio.Queue[tuple[str | None, Any]] = asyncio.Queue(maxsize=1)
        locks: dict[int, asyncio.Lock] = {}

        async def emit(name: str, payload: Any) -> None:
            """Queue one graph emission."""
            await queue.put((name, payload))

        async def run(name: str) -> None:
            """Run one root."""
            await self._execute_async(name, {"source": source, "data": source}, kwargs, None, locks, emit)

        async def report(error: BaseException | None) -> None:
            """Queue one root's failure, if any, followed by its completion marker."""
            if error is not None:
                await queue.put((None, error))
            await queue.put((None, _MISSING))

        notifications = []

        def finish(task: asyncio.Task) -> None:
            """Schedule completion reporting for one finished root."""
            if not task.cancelled():
                notifications.append(asyncio.create_task(report(task.exception())))

        tasks = [asyncio.create_task(run(name)) for name in self._roots()]
        for task in tasks:
            task.add_done_callback(finish)
        remaining = len(tasks)
        try:
            while remaining:
                name, payload = await queue.get()
                if name is not None:
                    yield name, payload
                elif payload is _MISSING:
                    remaining -= 1
                else:
                    raise payload
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for notification in notifications:
                notification.cancel()
            await asyncio.gather(*notifications, return_exceptions=True)

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

                if config.keys() & _CREDENTIAL_KEYS:
                    raise ValueError("LLM credentials cannot be stored in an Agent definition.")
                blocks[block_id] = LLM(**config)
            elif block_type == "Gate":
                blocks[block_id] = Gate(config)
            elif block_type in {"Dataset", "Image", "Export", "Deployment"}:
                if config.keys() & _CREDENTIAL_KEYS:
                    raise ValueError(f"{block_type} credentials cannot be stored in an Agent definition.")
                blocks[block_id] = globals()[block_type](**config)
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

    def _execute(self, name: str, event: dict[str, Any], kwargs: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        """Execute one Block and yield each emission before draining it downstream."""
        block = self.blocks[name]
        for emitted in self._invoke(name, event, kwargs):
            yield name, emitted.get(name, emitted["data"])
            downstream = (
                {key: value for key, value in emitted.items() if key != name} if isinstance(block, Gate) else emitted
            )
            for source, target in self.connections:
                if source == name:
                    yield from self._execute(target, {**downstream}, {})

    async def _execute_async(
        self,
        name: str,
        event: dict[str, Any],
        kwargs: dict[str, Any],
        outputs: dict[str, list[Any]] | None,
        locks: dict[int, asyncio.Lock],
        emit: Callable[[str, Any], Awaitable[None]] | None = None,
    ) -> None:
        """Execute one Block asynchronously and drain each emission downstream."""
        block = self.blocks[name]
        lock = locks.setdefault(id(block), asyncio.Lock())
        async for emitted in self._invoke_async(name, event, kwargs, lock):
            payload = emitted.get(name, emitted["data"])
            if outputs is not None:
                outputs[name].append(payload)
            if emit is not None:
                await emit(name, payload)
            downstream = (
                {key: value for key, value in emitted.items() if key != name} if isinstance(block, Gate) else emitted
            )
            tasks = [
                self._execute_async(target, {**downstream}, {}, outputs, locks, emit)
                for source, target in self.connections
                if source == name
            ]
            if tasks:
                await asyncio.gather(*tasks)

    def _invoke(self, name: str, event: dict[str, Any], kwargs: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Invoke one Block and normalize its emissions to Agent events."""
        block = self.blocks[name]
        if callable(agent_run := self._block_method(block, "_agent_run")):
            return self._events(agent_run(event, name, **kwargs))
        if inspect.iscoroutinefunction(block) or inspect.iscoroutinefunction(type(block).__call__):
            raise TypeError(f"Block {name!r} is asynchronous; use Agent.async_call().")
        output = block(event["data"], **kwargs)
        return () if output is None else ({**event, "data": output, name: output},)

    async def _invoke_async(
        self, name: str, event: dict[str, Any], kwargs: dict[str, Any], lock: asyncio.Lock
    ) -> AsyncIterator[dict[str, Any]]:
        """Invoke an async-capable Block without concurrently re-entering the same instance."""
        block = self.blocks[name]
        loop = asyncio.get_running_loop()
        if callable(agent_async_run := self._block_method(block, "_agent_async_run")):
            async with lock:
                output = await agent_async_run(event, name, **kwargs)
            for emitted in self._events(output):
                yield emitted
            return
        if callable(agent_run := self._block_method(block, "_agent_run")):
            async with lock:
                output = await loop.run_in_executor(None, partial(agent_run, event, name, **kwargs))
                iterator = iter(self._events(output))
                while True:
                    emitted = await loop.run_in_executor(None, self._next_event, iterator)
                    if emitted is _MISSING:
                        return
                    yield emitted
        elif callable(async_call := self._block_method(block, "async_call")):
            async with lock:
                output = await async_call(event["data"], **kwargs)
            if output is not None:
                yield {**event, "data": output, name: output}
        elif inspect.iscoroutinefunction(block) or inspect.iscoroutinefunction(type(block).__call__):
            async with lock:
                output = await block(event["data"], **kwargs)
            if output is not None:
                yield {**event, "data": output, name: output}
        else:
            async with lock:
                output = await loop.run_in_executor(None, partial(block, event["data"], **kwargs))
            if output is not None:
                yield {**event, "data": output, name: output}

    @staticmethod
    def _events(output: Any) -> Iterable[dict[str, Any]]:
        """Normalize one or many internal Block emissions to an iterable."""
        if output is None:
            return ()
        return (output,) if isinstance(output, Mapping) else output

    @staticmethod
    def _next_event(iterator: Iterator[dict[str, Any]]) -> dict[str, Any] | object:
        """Return the next event or a private end-of-stream sentinel."""
        return next(iterator, _MISSING)

    @staticmethod
    def _block_method(block: Callable[..., Any], name: str) -> Callable[..., Any] | None:
        """Return a Block protocol method without invoking dynamic attribute fallback."""
        method = inspect.getattr_static(block, name, None)
        return method.__get__(block, type(block)) if hasattr(method, "__get__") else method

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
        from ultralytics.models import LLM, RTDETR, YOLO, YOLOE, YOLOWorld

        if isinstance(block, LLM):
            config = {
                "model": block.model,
                "api": block.api,
                **{key: value for key, value in block.overrides.items() if key not in _CREDENTIAL_KEYS},
            }
            if block.base_url is not None:
                config["base_url"] = block.base_url
            if block.prompt is not None:
                config["prompt"] = block.prompt
            return {"type": "LLM", "config": config, "ports": deepcopy(block.ports)}
        if isinstance(block, (YOLO, YOLOWorld, YOLOE, RTDETR)):
            return {
                "type": "YOLO",
                "config": {"model": str(block.model_name), "task": block.task},
                "ports": deepcopy(block.ports),
            }
        if isinstance(block, Gate):
            return {"type": "Gate", "config": block.to_dict(), "ports": deepcopy(block.ports)}
        if isinstance(block, Dataset):
            return {
                "type": "Dataset",
                "config": {"source": block.source, "split": block.split},
                "ports": deepcopy(block.ports),
            }
        if isinstance(block, Image):
            return {"type": "Image", "config": {"source": block.source}, "ports": deepcopy(block.ports)}
        if isinstance(block, Export):
            return {
                "type": "Export",
                "config": {
                    "model": block.model,
                    "format": block.format,
                    **{key: value for key, value in block.overrides.items() if key not in _CREDENTIAL_KEYS},
                },
                "ports": deepcopy(block.ports),
            }
        if isinstance(block, Deployment):
            return {
                "type": "Deployment",
                "config": {"url": block.url},
                "ports": deepcopy(block.ports),
            }
        raise TypeError(f"Block type {type(block).__name__!r} is executable but not serializable.")


_MISSING = object()
