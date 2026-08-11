# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

from ultralytics.utils.checks import check_requirements


class LLM:
    """OpenAI-compatible large language model interface.

    Attributes:
        model (str): Model name sent with each request.
        api (str): API format, either "responses" or "chat.completions".
        client (OpenAI | None): Lazily initialized synchronous client.
        async_client (AsyncOpenAI | None): Lazily initialized asynchronous client.

    Methods:
        __call__: Run synchronous inference.
        async_call: Run asynchronous inference.

    Examples:
        Generate text with the Responses API:
        >>> from ultralytics import LLM
        >>> model = LLM("gpt-5.5")
        >>> response = model(input="What is YOLO?")
        >>> print(response.output_text)

        Analyze text and an image:
        >>> response = model(
        ...     input=[
        ...         {
        ...             "role": "user",
        ...             "content": [
        ...                 {"type": "input_text", "text": "What is in this image?"},
        ...                 {"type": "input_image", "image_url": "https://ultralytics.com/images/bus.jpg"},
        ...             ],
        ...         }
        ...     ]
        ... )

        Use the Chat Completions API:
        >>> model = LLM("gpt-5.5", api="chat.completions")
        >>> response = model(messages=[{"role": "user", "content": "What is YOLO?"}])
    """

    def __init__(
        self,
        model: str,
        api: str = "responses",
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an OpenAI-compatible LLM.

        Args:
            model (str): Model name.
            api (str): API format, either "responses" or "chat.completions".
            base_url (str, optional): OpenAI-compatible API base URL.
            api_key (str, optional): API key. Defaults to the OPENAI_API_KEY environment variable.
            **kwargs (Any): Additional arguments passed to the OpenAI client.
        """
        if api not in {"responses", "chat.completions"}:
            raise ValueError(f"Unsupported API format {api!r}. Use 'responses' or 'chat.completions'.")

        self.model = model
        self.api = api
        self.client = None
        self.async_client = None
        self._client_kwargs = {
            key: value
            for key, value in {"api_key": api_key, "base_url": base_url, **kwargs}.items()
            if value is not None
        }

    def __call__(self, **kwargs: Any) -> Any:
        """Run synchronous inference using native OpenAI request arguments."""
        request = {"model": self.model, **kwargs}
        client = self._get_client()
        return (
            client.responses.create(**request) if self.api == "responses" else client.chat.completions.create(**request)
        )

    async def async_call(self, **kwargs: Any) -> Any:
        """Run asynchronous inference using native OpenAI request arguments."""
        request = {"model": self.model, **kwargs}
        client = self._get_async_client()
        return (
            await client.responses.create(**request)
            if self.api == "responses"
            else await client.chat.completions.create(**request)
        )

    def _get_client(self) -> Any:
        """Create the OpenAI client on first inference."""
        if self.client is None:
            check_requirements("openai>=2.0.0")
            from openai import OpenAI

            self.client = OpenAI(**self._client_kwargs)
        return self.client

    def _get_async_client(self) -> Any:
        """Create the asynchronous OpenAI client on first inference."""
        if self.async_client is None:
            check_requirements("openai>=2.0.0")
            from openai import AsyncOpenAI

            self.async_client = AsyncOpenAI(**self._client_kwargs)
        return self.async_client
