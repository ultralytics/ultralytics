# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

from ultralytics.utils.checks import check_requirements


class LLM:
    """OpenAI-compatible large language model interface.

    Attributes:
        model (str): Model name sent with each request.
        api (str): API format, either "responses" or "chat.completions".
        base_url (str | None): Optional OpenAI-compatible API base URL.
        overrides (dict): Default arguments passed to each request.
        client (OpenAI | None): Lazily initialized synchronous client.
        async_client (AsyncOpenAI | None): Lazily initialized asynchronous client.

    Methods:
        __call__: Run synchronous inference.
        async_call: Run asynchronous inference.

    Examples:
        >>> from ultralytics import LLM
        >>> model = LLM("gpt-5.6-luna")
        >>> response = model("Describe this image")

        Use the Chat Completions API:
        >>> model = LLM("gpt-5.6-luna", api="chat.completions")
        >>> response = model("Describe this image")
    """

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
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
            **kwargs (Any): Default arguments passed to each API request.
        """
        if api not in {"responses", "chat.completions"}:
            raise ValueError(f"Unsupported API format {api!r}. Use 'responses' or 'chat.completions'.")

        self.model = model
        self.api = api
        self.base_url = base_url
        self.overrides = kwargs
        self.client = None
        self.async_client = None
        self._api_key = api_key

    def __call__(self, source: Any = None, **kwargs: Any) -> Any:
        """Run inference with the configured model."""
        request = self._request(source, kwargs)
        client = self._get_client()
        return (
            client.responses.create(**request) if self.api == "responses" else client.chat.completions.create(**request)
        )

    async def async_call(self, source: Any = None, **kwargs: Any) -> Any:
        """Run asynchronous inference with the configured model."""
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
        request = {**self.overrides, **kwargs, "model": self.model}
        if self.api == "responses":
            if source is not None:
                request["input"] = source
        elif source is not None:
            request["messages"] = [{"role": "user", "content": source}] if isinstance(source, str) else source
        return request

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
