---
comments: true
description: This guide provides best practices for performing thread-safe inference with YOLO models, ensuring reliable and concurrent predictions in multi-threaded applications.
keywords: thread-safe, YOLO inference, multi-threading, concurrent predictions, YOLO models, Ultralytics, Python threading, safe YOLO usage, AI concurrency
---

# Thread-Safe Inference with YOLO Models

Running YOLO models in a multi-threaded environment requires careful consideration to ensure thread safety. Python's `threading` module allows you to run several threads concurrently, but when it comes to using YOLO models across these threads, there are important safety issues to be aware of. This page will guide you through creating thread-safe YOLO model inference.

## Understanding Python Threading

Python threads are a form of parallelism that allow your program to run multiple operations at once. However, Python's Global Interpreter Lock (GIL) means that only one thread can execute Python bytecode at a time.

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/26833433/281418476-7f478570-fd77-4a40-bf3d-74b4db4d668c.png" alt="Single vs Multi-Thread Examples">
</p>

While this sounds like a limitation, threads can still provide concurrency, especially for I/O-bound operations or when using operations that release the GIL, like those performed by YOLO's underlying C libraries.

## The Danger of Shared Model Instances

Instantiating a YOLO model outside your threads and sharing this instance across multiple threads can lead to race conditions, where the internal state of the model is inconsistently modified due to concurrent accesses. This is particularly problematic when the model or its components hold state that is not designed to be thread-safe.

### Non-Thread-Safe Example: Single Model Instance

When using threads in Python, it's important to recognize patterns that can lead to concurrency issues. Here is what you should avoid: sharing a single YOLO model instance across multiple threads.

```python
# Unsafe: Sharing a single model instance across threads
from ultralytics import YOLO
from threading import Thread

# Instantiate the model outside the thread
shared_model = YOLO("yolov8n.pt")


def predict(image_path):
    results = shared_model.predict(image_path)
    # Process results


# Starting threads that share the same model instance
Thread(target=predict, args=("image1.jpg",)).start()
Thread(target=predict, args=("image2.jpg",)).start()
```

In the example above, the `shared_model` is used by multiple threads, which can lead to unpredictable results because `predict` could be executed simultaneously by multiple threads.

### Non-Thread-Safe Example: Multiple Model Instances

Similarly, here is an unsafe pattern with multiple YOLO model instances:

```python
# Unsafe: Sharing multiple model instances across threads can still lead to issues
from ultralytics import YOLO
from threading import Thread

# Instantiate multiple models outside the thread
shared_model_1 = YOLO("yolov8n_1.pt")
shared_model_2 = YOLO("yolov8n_2.pt")


def predict(model, image_path):
    results = model.predict(image_path)
    # Process results


# Starting threads with individual model instances
Thread(target=predict, args=(shared_model_1, "image1.jpg")).start()
Thread(target=predict, args=(shared_model_2, "image2.jpg")).start()
```

Even though there are two separate model instances, the risk of concurrency issues still exists. If the internal implementation of `YOLO` is not thread-safe, using separate instances might not prevent race conditions, especially if these instances share any underlying resources or states that are not thread-local.

## Thread-Safe Inference

To perform thread-safe inference, you should instantiate a separate YOLO model within each thread. This ensures that each thread has its own isolated model instance, eliminating the risk of race conditions.

### Thread-Safe Example

Here's how to instantiate a YOLO model inside each thread for safe parallel inference:

```python
# Safe: Instantiating a single model inside each thread
from ultralytics import YOLO
from threading import Thread


def thread_safe_predict(image_path):
    # Instantiate a new model inside the thread
    local_model = YOLO("yolov8n.pt")
    results = local_model.predict(image_path)
    # Process results


# Starting threads that each have their own model instance
Thread(target=thread_safe_predict, args=("image1.jpg",)).start()
Thread(target=thread_safe_predict, args=("image2.jpg",)).start()
```

In this example, each thread creates its own `YOLO` instance. This prevents any thread from interfering with the model state of another, thus ensuring that each thread performs inference safely and without unexpected interactions with the other threads.

## Conclusion

When using YOLO models with Python's `threading`, always instantiate your models within the thread that will use them to ensure thread safety. This practice avoids race conditions and makes sure that your inference tasks run reliably.

For more advanced scenarios and to further optimize your multi-threaded inference performance, consider using process-based parallelism with `multiprocessing` or leveraging a task queue with dedicated worker processes.
