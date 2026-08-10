# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

import cv2
import numpy as np
from PIL import Image

from ultralytics.utils.checks import check_requirements


class LLM:
    """OpenAI-compatible large language model interface.

    Attributes:
        model (str): Model name sent with each request.
        api (str): API format, either "responses" or "chat.completions".
        base_url (str | None): Optional OpenAI-compatible API base URL.
        prompt (str | None): Optional instruction prepended to scalar text or image inputs.
        overrides (dict): Default arguments passed to each request.
        client (OpenAI | None): Lazily initialized synchronous client.
        async_client (AsyncOpenAI | None): Lazily initialized asynchronous client.
        ports (dict): Agent input and output port definitions.

    Methods:
        __call__: Run synchronous inference.
        async_call: Run asynchronous inference.

    Examples:
        >>> from ultralytics import LLM
        >>> model = LLM("gpt-5.6-luna")
        >>> response = model("What is YOLO?")

        Analyze an image with a reusable prompt:
        >>> model = LLM("gpt-5.6-luna", prompt="Describe the image.")
        >>> response = model("path/to/image.jpg")

        Use the Chat Completions API:
        >>> model = LLM("gpt-5.6-luna", api="chat.completions")
        >>> response = model("Describe this image")
    """

    ports: ClassVar = {
        "inputs": {"source": "text | image", "context": "json"},
        "outputs": {"text": "text"},
    }

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        api: str = "responses",
        base_url: str | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an OpenAI-compatible LLM.

        Args:
            model (str): Model name.
            api (str): API format, either "responses" or "chat.completions".
            base_url (str, optional): OpenAI-compatible API base URL.
            api_key (str, optional): API key. Defaults to the OPENAI_API_KEY environment variable.
            prompt (str, optional): Instruction prepended to scalar text or image inputs.
            **kwargs (Any): Default arguments passed to each API request.
        """
        if api not in {"responses", "chat.completions"}:
            raise ValueError(f"Unsupported API format {api!r}. Use 'responses' or 'chat.completions'.")

        self.model = model
        self.api = api
        self.base_url = base_url
        self.prompt = prompt
        self.overrides = kwargs
        self.client = None
        self.async_client = None
        self._api_key = api_key

    def __call__(self, source: Any = None, **kwargs: Any) -> Any:
        """Run inference with the configured model."""
        return self._call(self._prepare(source), kwargs)

    def _call(self, source: Any, kwargs: dict[str, Any]) -> Any:
        """Send prepared input through the synchronous client."""
        request = self._request(source, kwargs)
        client = self._get_client()
        return (
            client.responses.create(**request) if self.api == "responses" else client.chat.completions.create(**request)
        )

    async def async_call(self, source: Any = None, **kwargs: Any) -> Any:
        """Run asynchronous inference with the configured model."""
        return await self._async_call(self._prepare(source), kwargs)

    async def _async_call(self, source: Any, kwargs: dict[str, Any]) -> Any:
        """Send prepared input through the asynchronous client."""
        request = self._request(source, kwargs)
        client = self._get_async_client()
        return (
            await client.responses.create(**request)
            if self.api == "responses"
            else await client.chat.completions.create(**request)
        )

    def _request(self, source: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build a Responses or Chat Completions request.

        Args:
            source (Any, optional): Responses input or chat messages. Strings become a user message for Chat
                Completions.
            kwargs (dict): Request arguments overriding constructor defaults.

        Returns:
            (dict): Native OpenAI SDK request arguments.
        """
        request = {"model": self.model, **self.overrides, **kwargs}
        if self.api == "responses":
            if source is not None:
                request["input"] = source
        elif source is not None:
            request["messages"] = [{"role": "user", "content": source}] if isinstance(source, str) else source
        return request

    def _prepare(self, source: Any) -> Any:
        """Normalize scalar text or image input while preserving native message payloads."""
        if source is None:
            return self.prompt
        if isinstance(source, (list, tuple, dict)):
            return source
        return self._agent_input({"source": source, "data": source})

    def _agent_run(self, event: dict[str, Any], name: str, **kwargs: Any) -> dict[str, Any]:
        """Run synchronous LLM inference on a standardized Agent event."""
        response = self._call(self._agent_input(event), kwargs)
        return {**event, "data": response, name: {"text": self._response_text(response)}}

    async def _agent_async_run(self, event: dict[str, Any], name: str, **kwargs: Any) -> dict[str, Any]:
        """Run asynchronous LLM inference on a standardized Agent event."""
        response = await self._async_call(self._agent_input(event), kwargs)
        return {**event, "data": response, name: {"text": self._response_text(response)}}

    def _agent_input(self, event: dict[str, Any]) -> Any:
        """Convert an Agent event into native Responses or Chat Completions input."""
        source = event["source"]
        context = {key: value for key, value in event.items() if key not in {"source", "data"}}
        prompt = self.prompt or "Describe the image."
        if context:
            prompt = f"{prompt}\n\nContext:\n{json.dumps(context, default=str)}"
        if source is None:
            return prompt
        if isinstance(source, str) and not self._is_image(source):
            return f"{prompt}\n\n{source}" if self.prompt or context else source
        image_url = self._image_url(source)
        if self.api == "responses":
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

    @staticmethod
    def _is_image(source: str) -> bool:
        """Return whether a string represents an image URL, data URL, or path."""
        suffixes = {
            ".bmp",
            ".gif",
            ".jpeg",
            ".jpg",
            ".png",
            ".tif",
            ".tiff",
            ".webp",
        }
        suffix = Path(source).suffix.lower()
        if source.startswith("data:image/"):
            return True
        url = urlsplit(source)
        if url.scheme in {"http", "https"}:
            return Path(url.path).suffix.lower() in suffixes
        return suffix in suffixes and Path(source).is_file()

    @staticmethod
    def _image_url(source: Any) -> str:
        """Convert an image URL, path, or array to an OpenAI image URL."""
        if isinstance(source, str) and source.startswith(("http://", "https://", "data:image/")):
            return source
        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
        else:
            image = (
                cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR)
                if isinstance(source, Image.Image)
                else np.asarray(source)
            )
        if image is None:
            raise ValueError(f"Unable to read Agent image source {source!r}.")
        _, buffer = cv2.imencode(".jpg", image)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

    @staticmethod
    def _response_text(response: Any) -> str:
        """Extract text from a Responses or Chat Completions result."""
        if isinstance(response, str):
            return response
        if text := getattr(response, "output_text", None):
            return text
        choices = getattr(response, "choices", None)
        return choices[0].message.content if choices else str(response)

    def _get_client(self) -> Any:
        """Create the OpenAI client on first inference."""
        if self.client is None:
            check_requirements("openai>=2.0.0")
            from openai import OpenAI

            kwargs = {k: v for k, v in {"api_key": self._api_key, "base_url": self.base_url}.items() if v is not None}
            self.client = OpenAI(**kwargs)
        return self.client

    def _get_async_client(self) -> Any:
        """Create the asynchronous OpenAI client on first inference."""
        if self.async_client is None:
            check_requirements("openai>=2.0.0")
            from openai import AsyncOpenAI

            kwargs = {k: v for k, v in {"api_key": self._api_key, "base_url": self.base_url}.items() if v is not None}
            self.async_client = AsyncOpenAI(**kwargs)
        return self.async_client
