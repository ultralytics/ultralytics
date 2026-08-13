---
comments: true
description: Use the Ultralytics LLM interface with OpenAI-compatible language and vision models through Responses or Chat Completions APIs.
keywords: Ultralytics LLM, OpenAI compatible API, Responses API, Chat Completions, vision language model, local LLM, YOLO
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

## Text and Images

### Text

```python
from ultralytics import LLM

model = LLM("gpt-5.6-luna")
response = model("What is YOLO and when should I use it?")
print(response.output_text)
```

Constructor keyword arguments become request defaults. Arguments passed to a call override them:

```python
model = LLM("gpt-5.6-luna", temperature=0.2, max_output_tokens=300)
response = model("Write a short YOLO26 release note.", max_output_tokens=120)
```

### Images

Pass a path, HTTP URL, data URI, NumPy array, or PIL image through `image`:

```python
from ultralytics import LLM

model = LLM("gpt-5.6-luna")
response = model("Describe this image.", image="https://ultralytics.com/images/bus.jpg")
print(response.output_text)
```

Local files and image objects are encoded as JPEG data URIs. NumPy arrays use OpenCV's BGR channel order. A string passed as `source`, such as `model("bus.jpg")`, is treated as text; use the `image` argument to send an image.

The optional `prompt` constructor argument prepends a reusable instruction to plain text and image requests:

```python
model = LLM("gpt-5.6-luna", prompt="Answer as a factory safety inspector in two sentences.")
response = model("Is the walkway clear?", image="aisle.jpg")
```

## Chat Completions

Select Chat Completions when required by the endpoint, and read its native response shape:

```python
from ultralytics import LLM

model = LLM("gpt-5.6-luna", api="chat.completions")
response = model("Explain non-maximum suppression.")
print(response.choices[0].message.content)
```

Pass a list of native message objects to preserve conversation history. Such lists are forwarded unchanged and do not receive the configured `prompt` prefix.

## Streaming and Async Calls

SDK request arguments pass through unchanged, including `stream=True`:

```python
model = LLM("gpt-5.6-luna")
for event in model("Explain object detection.", stream=True):
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

Use `async_call` for concurrent requests:

```python
import asyncio

from ultralytics import LLM

model = LLM("gpt-5.6-luna")


async def main():
    responses = await asyncio.gather(
        model.async_call("Describe this image.", image="image1.jpg"),
        model.async_call("Describe this image.", image="image2.jpg"),
    )
    return [response.output_text for response in responses]


print(asyncio.run(main()))
```

## OpenAI-Compatible Endpoints

Set `base_url` and the model identifier exposed by your provider. Many local servers implement Chat Completions, so check the server's API support:

```python
from ultralytics import LLM

model = LLM(
    "local-model",
    api="chat.completions",
    base_url="http://localhost:8000/v1",
    api_key="local",
)
response = model("Summarize the YOLO task list.")
print(response.choices[0].message.content)
```

Compatibility, model names, image support, and request arguments are determined by the selected provider. Use the provider's documentation rather than assuming every OpenAI-compatible server implements both APIs or every SDK option.

## Combine LLM and YOLO

YOLO can extract structured visual results before an LLM summarizes them:

```python
from collections import Counter

from ultralytics import LLM, YOLO

detector = YOLO("yolo26n.pt")
model = LLM("gpt-5.6-luna", prompt="Reply as a traffic analyst in two sentences.")

result = detector("https://ultralytics.com/images/bus.jpg")[0]
counts = Counter(result.names[int(cls)] for cls in result.boxes.cls)
response = model(f"Summarize these detections: {dict(counts)}")
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
