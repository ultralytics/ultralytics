from __future__ import annotations

from typing import Any

from ultralytics.utils.checks import check_requirements


class LLM:
    """OpenAI-compatible large language model interface.

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
        self._api_key = api_key

    def __call__(self, source: Any = None, **kwargs: Any) -> Any:
        """Run inference with the configured model."""
        return self.predict(source, **kwargs)

    def predict(self, source: Any = None, **kwargs: Any) -> Any:
        """Run inference using the Responses or Chat Completions API.

        Args:
            source (Any, optional): Responses input or chat messages. Strings become a user message for Chat
                Completions.
            **kwargs (Any): Request arguments overriding constructor defaults.

        Returns:
            (Any): Native OpenAI SDK response or stream.
        """
        request = {**self.overrides, **kwargs, "model": self.model}
        client = self._get_client()
        if self.api == "responses":
            if source is not None:
                request["input"] = source
            return client.responses.create(**request)

        if source is not None:
            request["messages"] = [{"role": "user", "content": source}] if isinstance(source, str) else source
        return client.chat.completions.create(**request)

    def _get_client(self) -> Any:
        """Create the OpenAI client on first inference."""
        if self.client is None:
            check_requirements("openai>=2.0.0")
            from openai import OpenAI

            kwargs = {k: v for k, v in {"api_key": self._api_key, "base_url": self.base_url}.items() if v is not None}
            self.client = OpenAI(**kwargs)
        return self.client
