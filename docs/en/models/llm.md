---
comments: true
description: Use the Ultralytics LLM interface with OpenAI, DeepSeek, Kimi, Z.AI GLM, OpenRouter, and other compatible providers.
keywords: Ultralytics LLM, OpenAI, DeepSeek, Kimi, Z.AI, GLM, OpenRouter, OpenAI compatible API, Responses API, Chat Completions, vision language model, local LLM, YOLO
---

# Ultralytics LLM Interface

`LLM` is a small interface for calling language and vision models through an OpenAI-compatible API. It prepares text or image input, forwards request arguments to the official OpenAI Python SDK, and returns the SDK's native response object. This makes it easy to add language or vision reasoning next to a [YOLO](yolo26.md) pipeline without wrapping provider-specific response types.

The default API is `responses`; use `api="chat.completions"` for providers that implement only Chat Completions.

## Installation

Install Ultralytics with the optional LLM dependency:

```bash
pip install "ultralytics[llm]"
```

For OpenAI, set the API key in the environment:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Responses API

The Responses API is the default. Pass a prompt and read `output_text`:

```python
from ultralytics import LLM

llm = LLM("gpt-5.6-luna")
response = llm("What is YOLO?")
print(response.output_text)
```

### Multimodal Input

Pass a path, HTTP URL, data URI, NumPy array, or PIL image through `image`:

```python
from ultralytics import LLM

llm = LLM("gpt-5.6-luna")
response = llm("What is happening in this image?", image="https://ultralytics.com/images/bus.jpg")
print(response.output_text)
```

Local files and image objects are encoded as JPEG data URIs. NumPy arrays use OpenCV's BGR channel order. A string passed as `source`, such as `llm("bus.jpg")`, is treated as text; use the `image` argument to send an image.

Use `prompt` for an instruction shared by every request:

```python
llm = LLM("gpt-5.6-luna", prompt="Answer in one sentence.")
response = llm("Describe this image.", image="bus.jpg")
```

## Chat Completions

Select Chat Completions when required by the endpoint, and read its native response shape:

```python
from ultralytics import LLM

llm = LLM("gpt-5.6-luna", api="chat.completions")
response = llm("What is non-maximum suppression?")
print(response.choices[0].message.content)
```

The same interface accepts multimodal input:

```python
response = llm("Describe this image.", image="bus.jpg")
print(response.choices[0].message.content)
```

Pass native message objects when you need conversation history. They are forwarded unchanged.

## Streaming and Async Calls

SDK request arguments pass through unchanged, including `stream=True`:

```python
llm = LLM("gpt-5.6-luna")
for event in llm("Explain object detection.", stream=True):
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

Use `async_call` for concurrent requests:

```python
import asyncio

from ultralytics import LLM

llm = LLM("gpt-5.6-luna")


async def main():
    response = await llm.async_call("What is YOLO?")
    print(response.output_text)


asyncio.run(main())
```

## OpenAI-Compatible Providers

`LLM` works with providers that expose an OpenAI-compatible Responses or Chat Completions API, including [DeepSeek](https://api-docs.deepseek.com/), [Kimi](https://platform.kimi.ai/docs/overview), [Z.AI GLM](https://docs.z.ai/guides/develop/openai/python), and [OpenRouter](https://openrouter.ai/docs/api_reference/overview). Set the provider's `base_url`, model name, and API key:

```python
from ultralytics import LLM

llm = LLM(
    "provider-model",
    api="chat.completions",
    base_url="https://provider.example/v1",
    api_key="your-api-key",
)
response = llm("What tasks does YOLO support?")
print(response.choices[0].message.content)
```

The same configuration works with compatible local servers. Model names, image support, request arguments, and supported APIs vary by provider, so check its linked documentation.

## Combine YOLO and an LLM

Run YOLO first, then call the LLM only when a person is detected:

```python
from ultralytics import LLM, YOLO

yolo = YOLO("yolo26n.pt")
llm = LLM("gpt-5.6-luna")
image = "https://ultralytics.com/images/bus.jpg"

result = yolo(image)[0]
# Call the LLM only when YOLO detects a person.
if any(result.names[int(cls)] == "person" for cls in result.boxes.cls):
    response = llm("Describe the scene.", image=image)
    print(response.output_text)
```

## API Reference

| Argument   | Default          | Description                                                                            |
| ---------- | ---------------- | -------------------------------------------------------------------------------------- |
| `model`    | `"gpt-5.6-luna"` | Model identifier sent to the selected endpoint                                         |
| `api`      | `"responses"`    | API format: `"responses"` or `"chat.completions"`                                      |
| `base_url` | `None`           | Optional OpenAI-compatible endpoint                                                    |
| `api_key`  | `None`           | API key; otherwise the SDK reads `OPENAI_API_KEY`                                      |
| `prompt`   | `None`           | Instruction prepended to plain text and image requests                                 |
| `**kwargs` |                  | Default SDK request arguments; per-call arguments override matching constructor values |

`source` accepts text, a native message list, or an image object. `image` accepts an image URL, data URI, path, NumPy array, or PIL image. Calling `model()` with a configured `prompt` sends the prompt by itself.

`LLM` is inference-only and is not exposed through the `yolo` CLI. It does not implement `train`, `val`, `export`, `track`, or `benchmark`.

## FAQ

### Which APIs does `LLM` support?

`LLM` uses the Responses API by default. Set `api="chat.completions"` for Chat Completions endpoints.

### Which providers can I use?

You can use OpenAI or another provider with a compatible API, such as [DeepSeek](https://api-docs.deepseek.com/), [Kimi](https://platform.kimi.ai/docs/overview), [Z.AI GLM](https://docs.z.ai/guides/develop/openai/python), or [OpenRouter](https://openrouter.ai/docs/api_reference/overview). Use the provider's model name, base URL, and API key.

### How do I send an image to a multimodal model?

Pass the prompt as the first argument and a local path, URL, data URI, NumPy array, or PIL image through `image`. Image support depends on the selected model and provider.

### Can I train or export an LLM with this class?

No. `LLM` provides synchronous and asynchronous inference only. Use Ultralytics `YOLO` for computer vision training, validation, prediction, export, tracking, and benchmarking.
