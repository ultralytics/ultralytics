---
comments: true
description: Reach any OpenAI-compatible large language or vision model from Ultralytics with the LLM interface. Works with OpenAI and other cloud providers as well as fully local on-prem servers.
keywords: Ultralytics LLM, OpenAI compatible API, vision language model, VLM, Responses API, Chat Completions, local LLM, on-prem LLM, Ollama, vLLM, LM Studio, llama.cpp, YOLO, async inference
---

# Ultralytics LLM: OpenAI-Compatible Large Language and Vision Model Interface

## Overview

`LLM` is a standalone [large language model](https://www.ultralytics.com/glossary/large-language-model-llm) interface shipped with the `ultralytics` package. It gives you a single, small class for talking to any OpenAI-compatible endpoint, so text reasoning and image understanding sit next to your [YOLO](yolo26.md) pipelines instead of in a separate codebase.

The class is deliberately thin. It builds the request, normalizes image inputs, and hands everything to the official OpenAI Python SDK. Responses come back as native SDK objects, and every keyword argument you pass is forwarded untouched, so nothing in the provider's feature set is hidden behind an Ultralytics abstraction.

Because the OpenAI API format has become the de facto standard for model serving, the same code reaches OpenAI's cloud, a hosted alternative provider, or a model running on your own hardware behind your own firewall. See [Local and On-Prem Deployment](#local-and-on-prem-deployment) for the full list.

### Key Features

- **One interface, many providers:** Point `base_url` at any OpenAI-compatible server. The code you write does not change when the provider does.
- **Both API formats:** Choose the modern `responses` API or the widely supported `chat.completions` API with a single argument.
- **Vision built in:** Pass a file path, URL, data URI, NumPy array, or PIL image and it is encoded and attached for you.
- **Native passthrough:** Constructor and per-call keyword arguments go straight to the SDK, so `temperature`, `stream`, `reasoning`, `tools`, `response_format`, and anything else your provider supports all work.
- **Sync and async:** Use `model(...)` for scripts and `await model.async_call(...)` for concurrent workloads.
- **Lazy and light:** The client is created on the first request, and `openai` is installed on demand rather than at import time.
- **On-prem ready:** Fully local, air-gapped inference is a `base_url` change, which keeps sensitive imagery inside your own network.

## Installation

The interface is part of the `ultralytics` package. Install the optional `llm` extra to pull in the OpenAI SDK:

```bash
pip install "ultralytics[llm]"
```

If `openai` is missing, Ultralytics installs it automatically on the first request.

Authentication uses the standard OpenAI environment variables, so no key is ever hardcoded in your scripts:

```bash
export OPENAI_API_KEY="sk-..." # required by the SDK, any non-empty value for most local servers
export OPENAI_BASE_URL="..."   # optional, overridden by the base_url argument
```

## Usage Examples

### Text Generation

The `responses` API is the default. Read the generated text from `response.output_text`.

!!! example "Text"

    === "Responses API"

        ```python
        from ultralytics import LLM

        # Load a model, the API key is read from OPENAI_API_KEY
        model = LLM("gpt-5.5")

        # Run inference
        response = model("What is YOLO and what is it used for?")
        print(response.output_text)
        ```

    === "Chat Completions API"

        ```python
        from ultralytics import LLM

        # Use the Chat Completions format instead
        model = LLM("gpt-5.5", api="chat.completions")

        response = model("What is YOLO and what is it used for?")
        print(response.choices[0].message.content)
        ```

### Image Understanding

Pass an image with the `image` argument to use any vision-capable model. A plain string `source` is always treated as text, so `model("bus.jpg")` sends the literal characters `bus.jpg` while `model("Describe this", image="bus.jpg")` sends the picture.

!!! example "Vision"

    === "Path or URL"

        ```python
        from ultralytics import LLM

        model = LLM("gpt-5.5")

        # From a local file, encoded and attached for you
        response = model("What is happening in this image?", image="path/to/bus.jpg")
        print(response.output_text)

        # From a URL, forwarded untouched for the provider to fetch
        response = model("Count the people and describe their clothing.", image="https://ultralytics.com/images/bus.jpg")
        print(response.output_text)
        ```

    === "Array or PIL image"

        ```python
        import cv2
        from PIL import Image

        from ultralytics import LLM

        model = LLM("gpt-5.5")

        # OpenCV BGR array
        im = cv2.imread("path/to/bus.jpg")
        print(model("Describe the scene.", image=im).output_text)

        # PIL image
        pil_im = Image.open("path/to/bus.jpg")
        print(model("Describe the scene.", image=pil_im).output_text)

        # An image passed on its own uses the default instruction "Describe the image."
        print(model(pil_im).output_text)
        ```

### Reusable Instructions

The `prompt` argument holds an instruction that is prepended to plain text and image requests. It keeps repeated framing out of every call site.

```python
from ultralytics import LLM

model = LLM("gpt-5.5", prompt="You are a factory safety inspector. Answer in at most two sentences.")

print(model("Is the operator wearing a helmet?", image="path/to/line.jpg").output_text)
print(model("Is the walkway clear?", image="path/to/aisle.jpg").output_text)
```

Native message payloads, meaning a list of message objects you build yourself, are sent exactly as written and `prompt` is not applied to them.

### Request Defaults and Overrides

Constructor keyword arguments become defaults for every request, and per-call keyword arguments override them. Both go directly to the OpenAI SDK.

```python
from ultralytics import LLM

# Defaults applied to every request
model = LLM("gpt-5.5", temperature=0.2, max_output_tokens=300)

# Override for a single request
response = model("Write a one paragraph release note for YOLO26.", max_output_tokens=120)
print(response.output_text)
```

### Multi-Turn Conversations

Pass a message list to keep conversation history. The list is forwarded as `input` for the `responses` API and as `messages` for `chat.completions`.

```python
from ultralytics import LLM

model = LLM("gpt-5.5", api="chat.completions")

messages = [
    {"role": "system", "content": "You are an Ultralytics deployment engineer."},
    {"role": "user", "content": "How do I export YOLO26 to TensorRT?"},
]

response = model(messages)
answer = response.choices[0].message.content
print(answer)

# Continue the conversation
messages.append({"role": "assistant", "content": answer})
messages.append({"role": "user", "content": "Now do it with INT8 quantization."})
print(model(messages).choices[0].message.content)
```

### Streaming

Streaming is a normal SDK argument, so pass `stream=True` and iterate the result.

!!! example "Streaming"

    === "Chat Completions API"

        ```python
        from ultralytics import LLM

        model = LLM("gpt-5.5", api="chat.completions")

        for chunk in model("Explain non-maximum suppression.", stream=True):
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
        ```

    === "Responses API"

        ```python
        from ultralytics import LLM

        model = LLM("gpt-5.5")

        for event in model("Explain non-maximum suppression.", stream=True):
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
        ```

### Asynchronous Inference

`async_call` mirrors `__call__` and uses an asynchronous client, which matters when you are captioning a folder of images or serving many requests at once.

```python
import asyncio

from ultralytics import LLM

model = LLM("gpt-5.5")


async def caption(paths):
    """Caption several images concurrently."""
    tasks = [model.async_call("Describe this image in one sentence.", image=p) for p in paths]
    return await asyncio.gather(*tasks)


captions = asyncio.run(caption(["im1.jpg", "im2.jpg", "im3.jpg"]))
for response in captions:
    print(response.output_text)
```

### Structured Output

Because arguments pass through untouched, JSON modes and schemas work as documented by your provider.

```python
import json

from ultralytics import LLM

model = LLM("gpt-5.5", api="chat.completions", response_format={"type": "json_object"})

response = model('List three YOLO tasks as JSON with a "tasks" array of strings.')
print(json.loads(response.choices[0].message.content))
```

## Local and On-Prem Deployment

The OpenAI API format is implemented by nearly every modern serving stack, so `LLM` runs against models hosted on your own machines with no code change beyond `base_url`. This is the right choice when imagery cannot leave the premises, when you need predictable latency, or when you want to avoid per-token costs on high-volume workloads. It also keeps [data privacy](https://www.ultralytics.com/glossary/data-privacy) and [data security](https://www.ultralytics.com/glossary/data-security) requirements satisfiable in air-gapped networks.

!!! example "Local inference"

    === "Ollama"

        ```python
        from ultralytics import LLM

        model = LLM(
            "qwen3-vl",
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # ignored by the server, required by the SDK
        )

        # Ollama serves the Responses API, so the default works unchanged
        print(model("Describe this image.", image="path/to/bus.jpg").output_text)
        ```

    === "vLLM"

        ```python
        from ultralytics import LLM

        # Server: vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
        model = LLM(
            "Qwen/Qwen3-VL-8B-Instruct",
            api="chat.completions",
            base_url="http://localhost:8000/v1",
            api_key="local",
        )

        print(model("What objects are visible?", image="path/to/bus.jpg").choices[0].message.content)
        ```

    === "LM Studio"

        ```python
        from ultralytics import LLM

        model = LLM(
            "local-model",
            api="chat.completions",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
        )

        print(model("Summarize the YOLO task list.").choices[0].message.content)
        ```

    === "llama.cpp"

        ```python
        from ultralytics import LLM

        # Server: llama-server -m model.gguf --port 8080
        model = LLM(
            "local-model",
            api="chat.completions",
            base_url="http://localhost:8080/v1",
            api_key="local",
        )

        print(model("Explain mAP in two sentences.").choices[0].message.content)
        ```

### Compatible Endpoints

The table lists common OpenAI-compatible servers and services with the base URL each one exposes by default. Anything else implementing the same routes works the same way.

| Runtime or service            | Typical `base_url`            | Deployment         | Notes                                                          |
| ----------------------------- | ----------------------------- | ------------------ | -------------------------------------------------------------- |
| OpenAI                        | default, no `base_url` needed | Cloud              | Supports both `responses` and `chat.completions`               |
| [Ollama](https://ollama.com/) | `http://localhost:11434/v1`   | Local              | Simple single-machine serving, serves both API formats         |
| vLLM                          | `http://localhost:8000/v1`    | Local, self-hosted | High-throughput GPU serving for production on-prem clusters    |
| SGLang                        | `http://localhost:30000/v1`   | Local, self-hosted | High-throughput serving with structured output support         |
| LM Studio                     | `http://localhost:1234/v1`    | Local              | Desktop app with a built-in compatible server                  |
| llama.cpp (`llama-server`)    | `http://localhost:8080/v1`    | Local              | GGUF serving for CPUs and consumer GPUs                        |
| LocalAI                       | `http://localhost:8080/v1`    | Local, self-hosted | Drop-in replacement covering several model backends            |
| Text Generation Inference     | `http://localhost:8080/v1`    | Self-hosted        | Hugging Face serving stack                                     |
| NVIDIA NIM                    | `http://localhost:8000/v1`    | Self-hosted        | Containerized microservices for on-prem GPUs                   |
| Azure OpenAI                  | your deployment endpoint      | Cloud              | Use the deployment name as `model`                             |
| Hosted alternatives           | provider endpoint             | Cloud              | Groq, Together, OpenRouter, Mistral, DeepSeek, xAI and similar |

!!! tip "On-prem checklist"

    - Many self-hosted servers implement `chat.completions` only, so pass `api="chat.completions"` unless the server documents Responses support.
    - The SDK requires a non-empty key even when the server ignores it. Pass any placeholder through `api_key` or set `OPENAI_API_KEY`.
    - Vision requests need a vision-capable checkpoint on the server, since images are sent as base64 JPEG data URIs.
    - Remote image URLs are forwarded untouched, and some servers refuse to fetch them. Ollama, for example, replies with `image URLs are not currently supported, please use base64 encoded data instead`. Pass a file path or an array instead and the image is encoded for you.
    - On reasoning models, a low token cap can be spent entirely on thinking, leaving the reply empty. Raise the cap or lower the reasoning effort if you get blank content.
    - Keep an eye on the server's context and image size limits. Large frames increase token usage on every request.

## Combining YOLO with LLM

The interface is most useful next to a detector. YOLO answers where and what, and the language model turns that into a description, a decision, or a report.

!!! example "Detection summary"

    === "Summarize detections"

        ```python
        from collections import Counter

        from ultralytics import LLM, YOLO

        detector = YOLO("yolo26n.pt")
        model = LLM("gpt-5.5", prompt="You are a traffic analyst. Reply in two sentences.")

        results = detector("path/to/bus.jpg")
        counts = Counter(results[0].names[int(c)] for c in results[0].boxes.cls)

        response = model(f"Describe this scene from the detected objects: {dict(counts)}")
        print(response.output_text)
        ```

    === "Verify with vision"

        ```python
        from ultralytics import LLM, YOLO

        detector = YOLO("yolo26n.pt")
        model = LLM("gpt-5.5")

        results = detector("path/to/bus.jpg")
        annotated = results[0].plot()  # BGR array with boxes drawn

        response = model("Are any of the labeled boxes clearly wrong?", image=annotated)
        print(response.output_text)
        ```

    === "Describe each crop"

        ```python
        from ultralytics import LLM, YOLO

        detector = YOLO("yolo26n.pt")
        model = LLM("gpt-5.5", prompt="Describe the object in one short phrase.")

        results = detector("path/to/bus.jpg")
        im = results[0].orig_img

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = im[y1:y2, x1:x2]
            print(results[0].names[int(box.cls)], model(image=crop).output_text)
        ```

For a fully local version of the same pipeline, run YOLO on your own hardware and point `LLM` at a self-hosted server, which keeps every frame and every prompt inside your network. This pairs naturally with [edge AI](https://www.ultralytics.com/glossary/edge-ai) deployments.

## Arguments

### Constructor Arguments

| Argument   | Type  | Default       | Description                                                                                         |
| ---------- | ----- | ------------- | --------------------------------------------------------------------------------------------------- |
| `model`    | `str` | `"gpt-5.5"`   | Model name sent with every request. For self-hosted servers, use the identifier the server exposes. |
| `api`      | `str` | `"responses"` | API format, either `"responses"` or `"chat.completions"`. Any other value raises `ValueError`.      |
| `base_url` | `str` | `None`        | OpenAI-compatible endpoint. Falls back to the SDK default or `OPENAI_BASE_URL`.                     |
| `api_key`  | `str` | `None`        | API key. Falls back to the `OPENAI_API_KEY` environment variable.                                   |
| `prompt`   | `str` | `None`        | Instruction prepended to plain text and image requests, ignored for native message payloads.        |
| `**kwargs` | `Any` |               | Default request arguments forwarded to the SDK, such as `temperature` or `max_output_tokens`.       |

### Call Arguments

| Argument   | Type  | Default | Description                                                                                            |
| ---------- | ----- | ------- | ------------------------------------------------------------------------------------------------------ |
| `source`   | `Any` | `None`  | Text string, a native list of message objects, or an image object. Strings are always treated as text. |
| `image`    | `Any` | `None`  | Image URL, `data:image/` URI, file path, `numpy.ndarray` (BGR), or PIL image. Encoded as base64 JPEG.  |
| `**kwargs` | `Any` |         | Per-request arguments merged over the constructor defaults and forwarded to the SDK.                   |

Both arguments are optional. Calling `model()` with a configured `prompt` sends that prompt on its own.

## Supported Modes

`LLM` is an inference-only client, not a trainable Ultralytics model. It has no `train`, `val`, `export`, or `benchmark` mode, and it is not available through the `yolo` CLI.

| Model | Tasks                        | Train | Val | Predict           | Export |
| ----- | ---------------------------- | ----- | --- | ----------------- | ------ |
| `LLM` | Text and image understanding | ❌    | ❌  | ✅ (sync + async) | ❌     |

## FAQ

### What is the Ultralytics LLM interface?

`LLM` is a small OpenAI-compatible client included in the `ultralytics` package. It sends text and images to any compatible endpoint and returns the provider's native response object. It exists so that language and vision-language reasoning can live in the same script as your YOLO inference, without a second dependency stack.

### Can I run it fully on-prem without sending data to OpenAI?

Yes. Set `base_url` to a local or internal server such as [Ollama](https://ollama.com/), vLLM, SGLang, LM Studio, llama.cpp, LocalAI, Text Generation Inference, or NVIDIA NIM, and the requests never leave your network. Ollama serves both API formats, so the default works there. Many of the others implement Chat Completions only, so pass `api="chat.completions"` when the Responses endpoint is missing. All of them still need a placeholder `api_key`, which the server usually ignores but the SDK requires.

### Should I use the Responses or the Chat Completions API?

Use `responses`, the default, with OpenAI and with providers that advertise Responses support, since it is the current format and exposes newer features. Use `chat.completions` for the widest compatibility, especially with self-hosted servers and third-party providers. The only difference in your code is how you read the reply: `response.output_text` for `responses` and `response.choices[0].message.content` for `chat.completions`.

### How do I send an image?

Use the `image` argument: `model("What is in this image?", image="bus.jpg")`. Paths, HTTP URLs, `data:image/` URIs, NumPy arrays, and PIL images are all accepted. URLs and data URIs are passed through as-is, while everything else is encoded to a base64 JPEG. NumPy arrays are read as BGR, matching OpenCV and `Results.plot()`. Passing a bare string as `source` sends text, never an image. Some self-hosted servers reject remote URLs and accept only base64 data, so prefer a path or an array there.

### Does it support streaming, async, and tool calling?

Yes to all three, because keyword arguments reach the SDK unchanged. Pass `stream=True` and iterate the result, use `await model.async_call(...)` for concurrency, and supply `tools` or `response_format` exactly as your provider documents them.

### Can I train, fine-tune, or export a model through this interface?

No. `LLM` is a client for models that are already served elsewhere. Training, validation, and export in Ultralytics apply to vision models such as [YOLO26](yolo26.md), documented under [Train](../modes/train.md), [Val](../modes/val.md), and [Export](../modes/export.md). To fine-tune a language model, train it with your provider or serving stack and then point `model` at the resulting checkpoint name.

### Which package extras do I need?

Install with `pip install "ultralytics[llm]"`, which adds `openai>=2.0.0`. If the SDK is missing at runtime, Ultralytics installs it on the first request.
