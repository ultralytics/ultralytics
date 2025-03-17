---
comments: true
description: Learn how to ensure thread-safe YOLO model inference in Python. Avoid race conditions and run your multi-threaded tasks reliably with best practices.
keywords: YOLO models, thread-safe, Python threading, model inference, concurrency, race conditions, multi-threaded, parallelism, Python GIL
---

# Thread-Safe Inference with YOLO Models

Running YOLO models in a multi-threaded environment requires careful consideration to ensure thread safety. Python's `threading` module allows you to run several threads concurrently, but when it comes to using YOLO models across these threads, there are important safety issues to be aware of. This page will guide you through creating thread-safe YOLO model inference.

## Understanding Python Threading

Python threads are a form of parallelism that allow your program to run multiple operations at once. However, Python's Global Interpreter Lock (GIL) means that only one thread can execute Python bytecode at a time.

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/single-vs-multi-thread-examples.avif" alt="Single vs Multi-Thread Examples">
</p>

While this sounds like a limitation, threads can still provide concurrency, especially for I/O-bound operations or when using operations that release the GIL, like those performed by YOLO's underlying C libraries.

## The Danger of Shared Model Instances

Instantiating a YOLO model outside your threads and sharing this instance across multiple threads can lead to [race conditions](https://www.ultralytics.com/glossary/algorithmic-bias), where the internal state of the model is inconsistently modified due to concurrent accesses. This is particularly problematic when the model or its components hold state that is not designed to be thread-safe.

### Non-Thread-Safe Example: Single Model Instance

When using threads in Python, it's important to recognize patterns that can lead to concurrency issues. Here is what you should avoid: sharing a single YOLO model instance across multiple threads.

```python
# Unsafe: Sharing a single model instance across threads
from threading import Thread

from ultralytics import YOLO

# Instantiate the model outside the thread
shared_model = YOLO("yolo11n.pt")


def predict(image_path):
    """Predicts objects in an image using a preloaded YOLO model, take path string to image as argument."""
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
from threading import Thread

from ultralytics import YOLO

# Instantiate multiple models outside the thread
shared_model_1 = YOLO("yolo11n_1.pt")
shared_model_2 = YOLO("yolo11n_2.pt")


def predict(model, image_path):
    """Runs prediction on an image using a specified YOLO model, returning the results."""
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
from threading import Thread

from ultralytics import YOLO


def thread_safe_predict(image_path):
    """Predict on an image using a new YOLO model instance in a thread-safe manner; takes image path as input."""
    local_model = YOLO("yolo11n.pt")
    results = local_model.predict(image_path)
    # Process results


# Starting threads that each have their own model instance
Thread(target=thread_safe_predict, args=("image1.jpg",)).start()
Thread(target=thread_safe_predict, args=("image2.jpg",)).start()
```

In this example, each thread creates its own `YOLO` instance. This prevents any thread from interfering with the model state of another, thus ensuring that each thread performs inference safely and without unexpected interactions with the other threads.

## Using ThreadingLocked Decorator

Ultralytics provides a `ThreadingLocked` decorator that can be used to ensure thread-safe execution of functions. This decorator uses a lock to ensure that only one thread at a time can execute the decorated function.

```python
from ultralytics import YOLO
from ultralytics.utils import ThreadingLocked

# Create a model instance
model = YOLO("yolo11n.pt")


# Decorate the predict method to make it thread-safe
@ThreadingLocked()
def thread_safe_predict(image_path):
    """Thread-safe prediction using a shared model instance."""
    results = model.predict(image_path)
    return results


# Now you can safely call this function from multiple threads
```

The `ThreadingLocked` decorator is particularly useful when you need to share a model instance across threads but want to ensure that only one thread can access it at a time. This approach can save memory compared to creating a new model instance for each thread, but it may reduce concurrency as threads will need to wait for the lock to be released.

## Conclusion

When using YOLO models with Python's `threading`, always instantiate your models within the thread that will use them to ensure thread safety. This practice avoids race conditions and makes sure that your inference tasks run reliably.

For more advanced scenarios and to further optimize your multi-threaded inference performance, consider using process-based parallelism with [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) or leveraging a task queue with dedicated worker processes.

## FAQ

### How can I avoid race conditions when using YOLO models in a multi-threaded Python environment?

To prevent race conditions when using Ultralytics YOLO models in a multi-threaded Python environment, instantiate a separate YOLO model within each thread. This ensures that each thread has its own isolated model instance, avoiding concurrent modification of the model state.

Example:

```python
from threading import Thread

from ultralytics import YOLO


def thread_safe_predict(image_path):
    """Predict on an image in a thread-safe manner."""
    local_model = YOLO("yolo11n.pt")
    results = local_model.predict(image_path)
    # Process results


Thread(target=thread_safe_predict, args=("image1.jpg",)).start()
Thread(target=thread_safe_predict, args=("image2.jpg",)).start()
```

For more information on ensuring thread safety, visit the [Thread-Safe Inference with YOLO Models](#thread-safe-inference).

### What are the best practices for running multi-threaded YOLO model inference in Python?

To run multi-threaded YOLO model inference safely in Python, follow these best practices:

1. Instantiate YOLO models within each thread rather than sharing a single model instance across threads.
2. Use Python's `multiprocessing` module for parallel processing to avoid issues related to Global Interpreter Lock (GIL).
3. Release the GIL by using operations performed by YOLO's underlying C libraries.
4. Consider using the `ThreadingLocked` decorator for shared model instances when memory is a concern.

Example for thread-safe model instantiation:

```python
from threading import Thread

from ultralytics import YOLO


def thread_safe_predict(image_path):
    """Runs inference in a thread-safe manner with a new YOLO model instance."""
    model = YOLO("yolo11n.pt")
    results = model.predict(image_path)
    # Process results


# Initiate multiple threads
Thread(target=thread_safe_predict, args=("image1.jpg",)).start()
Thread(target=thread_safe_predict, args=("image2.jpg",)).start()
```

For additional context, refer to the section on [Thread-Safe Inference](#thread-safe-inference).

### Why should each thread have its own YOLO model instance?

Each thread should have its own YOLO model instance to prevent race conditions. When a single model instance is shared among multiple threads, concurrent accesses can lead to unpredictable behavior and modifications of the model's internal state. By using separate instances, you ensure thread isolation, making your multi-threaded tasks reliable and safe.

For detailed guidance, check the [Non-Thread-Safe Example: Single Model Instance](#non-thread-safe-example-single-model-instance) and [Thread-Safe Example](#thread-safe-example) sections.

### How does Python's Global Interpreter Lock (GIL) affect YOLO model inference?

Python's Global Interpreter Lock (GIL) allows only one thread to execute Python bytecode at a time, which can limit the performance of CPU-bound multi-threading tasks. However, for I/O-bound operations or processes that use libraries releasing the GIL, like YOLO's underlying C libraries, you can still achieve concurrency. For enhanced performance, consider using process-based parallelism with Python's `multiprocessing` module.

For more about threading in Python, see the [Understanding Python Threading](#understanding-python-threading) section.

### Is it safer to use process-based parallelism instead of threading for YOLO model inference?

Yes, using Python's `multiprocessing` module is safer and often more efficient for running YOLO model inference in parallel. Process-based parallelism creates separate memory spaces, avoiding the Global Interpreter Lock (GIL) and reducing the risk of concurrency issues. Each process will operate independently with its own YOLO model instance.

For further details on process-based parallelism with YOLO models, refer to the page on [Thread-Safe Inference](#thread-safe-inference).
