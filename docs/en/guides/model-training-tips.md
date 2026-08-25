---
title: Model Training Tips & Best Practices
comments: true
description: Train YOLO computer vision models more efficiently with proven tips on batch size, mixed precision, caching, early stopping, and optimizer choice.
keywords: Model Training Machine Learning, AI Model Training, Number of Epochs, How to Train a Model in Machine Learning, Machine Learning Best Practices, What is Model Training
---

# Machine Learning Best Practices and Tips for Model Training

Training an Ultralytics YOLO model well comes down to a handful of settings: batch size, caching, mixed precision, the number of epochs, early stopping, and the optimizer. This guide covers what each one does, the value to start from, and how to recognize when it needs changing, so you train faster on the hardware you already have and stop runs that have stopped improving.

[Model training](../modes/train.md) is the process of teaching your model to recognize visual patterns and make predictions from your data, and it directly shapes the accuracy of your application. It comes after you [define your project goals](./defining-project-goals.md), [collect and annotate your data](./data-collection-and-annotation.md), and [preprocess the annotations](./preprocessing-annotated-data.md) — see [steps of a computer vision project](./steps-of-a-cv-project.md) for the full pipeline.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GIrFEoR5PoU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Model Training Tips | How to Handle Large Datasets | Batch Size, GPU Utilization and <a href="https://www.ultralytics.com/glossary/mixed-precision">Mixed Precision</a>
</p>

!!! example "The settings this guide covers"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # pretrained weights (transfer learning)
        model.train(
            data="coco8.yaml",
            epochs=300,  # long budget; early stopping ends it sooner
            cache=True,  # cache images in RAM to keep the GPU fed
            patience=50,  # stop when validation stops improving
            optimizer="auto",  # AdamW or MuSGD, chosen from the iteration count
        )
        ```

    === "CLI"

        ```bash
        yolo detect train model=yolo26n.pt data=coco8.yaml epochs=300 cache=True patience=50 optimizer=auto
        ```

## How to Train a Machine Learning Model

A computer vision model is trained by adjusting its internal parameters to minimize errors. Initially, the model is fed a large set of labeled images. It makes predictions about what is in these images, and the predictions are compared to the actual labels or contents to calculate errors. These errors show how far off the model's predictions are from the true values.

During training, the model iteratively makes predictions, calculates errors, and updates its parameters through a process called [backpropagation](https://www.ultralytics.com/glossary/backpropagation). In this process, the model adjusts its internal parameters (weights and biases) to reduce the errors. By repeating this cycle many times, the model gradually improves its accuracy. Over time, it learns to recognize complex patterns such as shapes, colors, and textures.

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/e383fb3070bc3af833dbaf5cf5f5e2b6.avif" alt="Backpropagation loop: forward pass, error calculation, and weight update">
</p>

This learning process makes it possible for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) model to perform various [tasks](../tasks/index.md), including [object detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), and [image classification](../tasks/classify.md). The ultimate goal is to create a model that can generalize its learning to new, unseen images so that it can accurately understand visual data in real-world applications.

## Training on Large Datasets

Large datasets stress GPU memory and disk I/O before they stress the model. Five Ultralytics YOLO settings control that tradeoff: `batch`, `fraction`, `multi_scale`, `cache`, and `amp`.

### Batch Size and GPU Utilization

When training models on large datasets, efficiently utilizing your GPU is key. Batch size is an important factor. It is the number of data samples that a machine learning model processes in a single training iteration.
Using the maximum batch size supported by your GPU, you can fully take advantage of its capabilities and reduce the time model training takes. However, you want to avoid running out of GPU memory. If you encounter memory errors, reduce the batch size incrementally until the model trains smoothly.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gxl6Bbpcxs0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Use Batch Inference with Ultralytics YOLO26 | Speed Up Object Detection in Python 🎉
</p>

With respect to YOLO26, you can set the `batch` parameter in the [Train mode arguments](../modes/train.md) to match your GPU capacity. On a CUDA GPU, `batch=-1` profiles the model and picks a [batch size](https://www.ultralytics.com/glossary/batch-size) that fills roughly 60% of GPU memory, and a fraction such as `batch=0.70` targets a different share instead. On CPU and Apple silicon it is a no-op that warns and falls back to the default `batch=16`, because there is no GPU memory to profile.

Batch size alone will not saturate a GPU. If utilization oscillates between full and near-zero instead of holding steady, the bottleneck is the input pipeline rather than the model: the GPU drains a batch faster than the CPU can decode and augment the next one. The `workers` parameter sets how many dataloader processes feed each device (8 by default, and per rank under multi-GPU), so raise it toward `os.cpu_count()` divided by the number of devices, which is the ceiling the dataloader enforces, and combine it with [caching](#caching). On `device=cpu` and `device=mps` the trainer forces `workers=0`, because there the model, not the loader, is the limit.

### Subset Training

Subset training trains your model on a smaller slice of the dataset. It can save time and resources, especially during initial model development and testing. If you are running short on time or experimenting with different model configurations, subset training is a good option.

When it comes to YOLO26, you can implement subset training with the `fraction` parameter, which sets what share of the training split to use. Setting `fraction=0.1` keeps the first 10% of the sorted image list — a deterministic head slice rather than a stratified sample, so class balance is not preserved — and it applies to the training split only. Shuffle or pre-split the dataset yourself when balance matters. When the full dataset is itself small, [K-fold cross-validation](./kfold-cross-validation.md) gives a more reliable accuracy estimate than a single train/val split.

!!! tip "Fix the seed before you compare"

    A subset run is only worth comparing against another if the setting under test is the only thing that changed. `seed=0` and `deterministic=True` are the defaults, so the main risk is overriding them without noticing, and `cache='ram'` gives up determinism on its own. Change one argument per run and give each run a distinct `name`, so the results land in directories you can still tell apart a week later.

### Multi-scale Training

Multi-scale training is a technique that improves your model's ability to generalize by training it on images of varying sizes. Your model can learn to detect objects at different scales and distances and become more robust.

For example, when you train YOLO26, you can enable scale augmentation by setting the `scale` parameter. This parameter adjusts the size of training images by a specified factor, simulating objects at different distances. For example, setting `scale=0.5` randomly zooms training images by a factor between 0.5 and 1.5 during training. Configuring this parameter allows your model to experience a variety of image scales and improve its detection capabilities across different object sizes and scenarios.

Ultralytics also supports image-size multi-scale training via the `multi_scale` parameter. Unlike `scale`, which zooms images and then pads/crops back to `imgsz`, `multi_scale` changes `imgsz` itself each batch (rounded to the model stride). For example, with `imgsz=640` and `multi_scale=0.25`, the training size is sampled from 480 up to 800 in stride steps (e.g., 480, 512, 544, ..., 800), while `multi_scale=0.0` keeps a fixed size.

### Caching

Caching is an important technique to improve the efficiency of training machine learning models. By storing preprocessed images in memory, caching reduces the time the GPU spends waiting for data to be loaded from the disk. The model can continuously receive data without delays caused by disk I/O operations.

Caching can be controlled when training YOLO26 using the `cache` parameter:

| Value          | Where images are stored | Tradeoff                                       |
| -------------- | ----------------------- | ---------------------------------------------- |
| `cache=True`   | RAM                     | Fastest access, highest memory use             |
| `cache='disk'` | Local disk              | Slower than RAM, faster than re-reading source |
| `cache=False`  | Not cached (default)    | No extra memory, slowest data loading          |

!!! warning "Caching is not free"

    `cache=True` holds every decoded training image in RAM at once. Ultralytics estimates the requirement from a sample of images, adds a safety margin, and falls back to no caching when that does not fit in available memory, logging a warning as it does — so a run that appears to ignore `cache=True` is usually a run that did not fit, and the warning in the log says so. Two more things to plan for:

    - **Multi-GPU**: every DDP rank builds its own dataset and its own cache, so N GPUs hold N copies of the dataset in system RAM. The memory check runs per rank and cannot see the other ranks.
    - **Reproducibility**: `cache='ram'` can produce non-deterministic results and warns accordingly. Use `cache='disk'` when runs need to be repeatable.

### Mixed Precision Training

Mixed precision training uses both 16-bit (FP16) and 32-bit (FP32) floating-point types. The strengths of both FP16 and FP32 are leveraged by using FP16 for faster computation and FP32 to maintain precision where needed. Most of the [neural network](https://www.ultralytics.com/glossary/neural-network-nn)'s operations are done in FP16 to benefit from faster computation and lower memory usage. However, a master copy of the model's weights is kept in FP32 to ensure accuracy during the weight update steps. You can handle larger models or larger batch sizes within the same hardware constraints.

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/886e0abf0da10bc0b3d6a6c7b06bb9e1.avif" alt="Mixed precision FP16 training benefits">
</p>

To implement mixed precision training, you'll need to modify your training scripts and ensure your hardware (like GPUs) supports it. Many modern [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) frameworks, such as [PyTorch](https://www.ultralytics.com/glossary/pytorch) and [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), offer built-in support for mixed precision.

Mixed precision training is straightforward when working with YOLO26: AMP is already on by default (`amp=True`). On a CUDA GPU, YOLO runs a one-time capability check at the start of training and falls back to full FP32 if it fails; on CPU and Apple silicon AMP is skipped entirely. Set `amp=False` to force FP32.

### Pretrained Weights

Pretrained weights shorten training by starting from features already learned on a large dataset. [Transfer learning](https://www.ultralytics.com/glossary/transfer-learning) adapts pretrained models to new, related tasks. Fine-tuning a pretrained model involves starting with these weights and then continuing training on your specific dataset. This method of training results in faster training times and often better performance because the model starts with a solid understanding of basic features.

Pretrained weights come from the `model` argument, so `model=yolo26n.pt` already starts from COCO weights. The `pretrained` parameter controls what happens to them: keep them (`True`, the default), discard them and train from scratch (`False`), or load a different checkpoint into the architecture (`pretrained=path/to/best.pt`, which is how you transfer weights into a `model=*.yaml` build). Setting `pretrained=True` on a `*.yaml` model does not fetch weights on its own. To push a small model further than pretrained weights alone allow, train it against a larger one with [knowledge distillation](./knowledge-distillation.md), and see [How to Fine-Tune YOLO on a Custom Dataset](./finetuning-guide.md) for the full transfer-learning workflow.

### Other Techniques to Consider When Handling a Large Dataset

Four further settings matter once the dataset outgrows a single GPU or a single run:

- **[Learning Rate](https://www.ultralytics.com/glossary/learning-rate) Schedulers**: Implementing learning rate schedulers dynamically adjusts the learning rate during training. A well-tuned learning rate can prevent the model from overshooting minima and improve stability. When training YOLO26, the `lrf` parameter helps manage learning rate scheduling by setting the final learning rate as a fraction of the initial rate. Class imbalance has its own argument, `cls_pw`, which applies inverse-frequency class weighting without any custom code. For behavior no argument exposes, such as per-layer learning rates or gradient clipping, [subclass the trainer](./custom-trainer.md).
- **Distributed Training**: For handling large datasets, distributed training can be a game-changer. You can reduce the training time by spreading the training workload across multiple GPUs or machines. This approach is particularly valuable for enterprise-scale projects with substantial computational resources.
- **Graph Compilation**: `compile=True` compiles the model with the PyTorch `inductor` backend, trading a one-off compilation at the start of the run for faster steps afterwards. It pays for itself on long runs and falls back to eager execution with a warning wherever it is unsupported.
- **channels_last Memory Format**: `channels_last=True` runs convolutions in NHWC, which maps better onto Tensor Cores on modern CUDA GPUs. It changes throughput only, never results, and is ignored on CPU and Apple silicon.

## The Number of Epochs To Train For

When training a model, an [epoch](https://www.ultralytics.com/glossary/epoch) refers to one complete pass through the entire training dataset. During an epoch, the model processes each example in the training set once and updates its parameters based on the learning algorithm. Multiple epochs are usually needed to allow the model to learn and refine its parameters over time.

A common question that comes up is how to determine the number of epochs to train the model for. The default is `epochs=100`, but 300 is a better starting point for a real dataset. If the model overfits early, you can reduce the number of epochs. If [overfitting](https://www.ultralytics.com/glossary/overfitting) does not occur after 300 epochs, you can extend the training to 600, 1200, or more epochs.

However, the ideal number of epochs can vary based on your dataset's size and project goals. Larger datasets might require more epochs for the model to learn effectively, while smaller datasets might need fewer epochs to avoid overfitting. With respect to YOLO26, you can set the `epochs` parameter in your training script.

### Training on a Time Budget

Epochs are a proxy for compute, but hours are what a rented GPU bills and what a cluster job limit cuts off. The `time` parameter caps training in wall-clock hours and overrides `epochs` when set: after the first epoch the trainer estimates how many epochs fit in the budget, then stops on the budget rather than on a count. The budget is checked every batch, so a run can stop part-way through an epoch — but it stops gracefully, validating and writing `best.pt` on the way out rather than being killed outright.

```bash
yolo train model=yolo26n.pt data=coco8.yaml time=6
```

## Early Stopping

Early stopping is a valuable technique for optimizing model training. By monitoring validation performance, you can halt training once the model stops improving. You can save computational resources and prevent overfitting.

The process involves setting a patience parameter that determines how many epochs to wait for an improvement in validation metrics before stopping training. If the model's performance does not improve within these epochs, training is stopped to avoid wasting time and resources.

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/a4a45c9d1d0409cc45ba1416184559dc.avif" alt="Early stopping to prevent model overfitting">
</p>

For YOLO26, you can enable early stopping by setting the patience parameter in your training configuration. For example, `patience=5` means training will stop if there's no improvement in validation metrics for 5 consecutive epochs. Using this method ensures the training process remains efficient and achieves optimal performance without excessive computation.

## Choosing Between Cloud and Local Training

YOLO models train equally well in the cloud and on local hardware; the choice is a cost, control, and data-residency tradeoff rather than a capability one.

Cloud training offers scalability and powerful hardware and is ideal for handling large datasets and complex models. Platforms like [Google Cloud](https://cloud.google.com/), [AWS](https://aws.amazon.com/), and [Azure](https://azure.microsoft.com/) provide on-demand access to high-performance GPUs and TPUs, speeding up training times and enabling experiments with larger models. However, cloud training can be expensive, especially for long periods, and data transfer can add to costs and latency.

The cheapest cloud GPUs are spot and preemptible instances, which can be reclaimed at any moment. That is workable as long as the run is checkpointed on a cadence you can afford to lose: `last.pt` is written at the end of every epoch, and `save_period=N` adds numbered `epoch{N}.pt` snapshots, which matter when a single epoch is long. Restart the reclaimed job against the same run directory with `resume=True` and it picks up the weights, optimizer state, and epoch counter. See [Resuming Interrupted Trainings](../modes/train.md#resuming-interrupted-trainings).

Local training provides greater control and customization, letting you tailor your environment to specific needs and avoid ongoing cloud costs. It can be more economical for long-term projects, and since your data stays on-premises, it's more secure. However, local hardware may have resource limitations and require maintenance, which can lead to longer training times for large models.

## Selecting an Optimizer

An optimizer is an algorithm that adjusts the weights of your neural network to minimize the [loss function](https://www.ultralytics.com/glossary/loss-function), which measures how well the model is performing. In simpler terms, the optimizer helps the model learn by tweaking its parameters to reduce errors. Choosing the right optimizer directly affects how quickly and accurately the model learns.

You can also fine-tune optimizer parameters to improve model performance. Adjusting the learning rate sets the size of the steps when updating parameters. For stability, you might start with a moderate learning rate and gradually decrease it over time to improve long-term learning. Additionally, setting the momentum determines how much influence past updates have on current updates. A common value for momentum is around 0.9. It generally provides a good balance.

### Common Optimizers

Ultralytics YOLO26 supports SGD, MuSGD, Adam, Adamax, AdamW, NAdam, RAdam, and RMSProp. Their tradeoffs:

| Optimizer                                                                                    | How it updates weights                                                                                                               | Best for                                                                                                                             |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| **SGD** (Stochastic Gradient Descent)                                                        | Applies the raw gradient of the loss with respect to each parameter                                                                  | Simple, memory-light runs; can converge slowly and stall in local minima                                                             |
| **[Adam](https://www.ultralytics.com/glossary/adam-optimizer)** (Adaptive Moment Estimation) | Adapts the learning rate per parameter from first- and second-moment estimates                                                       | Noisy data and sparse gradients; needs little tuning                                                                                 |
| **AdamW**                                                                                    | Adam with decoupled weight decay                                                                                                     | Short runs — `optimizer=auto` selects AdamW at 10,000 optimizer steps or fewer                                                       |
| **RMSProp** (Root Mean Square Propagation)                                                   | Divides the gradient by a running average of recent gradient magnitudes                                                              | Vanishing-gradient regimes and [RNNs](https://www.ultralytics.com/glossary/recurrent-neural-network-rnn)                             |
| **MuSGD** (Muon + SGD hybrid)                                                                | Orthogonalizes gradients by Newton-Schulz iteration for weight matrices and conv filters, blended with SGD; biases and norms use SGD | Long, large-scale runs; `optimizer=auto` selects it past 10,000 steps. See the [YOLO26 training recipe](./yolo26-training-recipe.md) |

For YOLO26, the `optimizer` parameter accepts SGD, MuSGD, Adam, Adamax, AdamW, NAdam, RAdam, and RMSProp, or `auto` to select one from the model configuration and the total iteration count:

!!! example "Set the optimizer explicitly"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.train(data="coco8.yaml", optimizer="MuSGD")
        ```

    === "CLI"

        ```bash
        yolo train model=yolo26n.pt data=coco8.yaml optimizer=MuSGD
        ```

## Conclusion

Efficient Ultralytics YOLO training comes down to a few settings: size `batch` to your GPU, turn on `cache` and keep `amp` enabled so the GPU stays fed, start from pretrained weights, and let `patience` end a run that has converged. Set a generous `epochs` budget, or cap the run with `time`, and let early stopping decide when to finish. From here, search the settings automatically with the [hyperparameter tuning guide](./hyperparameter-tuning.md), or start from the published defaults in the [YOLO26 training recipe](./yolo26-training-recipe.md). If questions come up along the way, ask the community on the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics/issues) or the [Ultralytics Discord server](https://discord.com/invite/ultralytics).

## FAQ

### How can I improve GPU utilization when training a large dataset with Ultralytics YOLO?

To improve GPU utilization, set the `batch` parameter in your training configuration to the maximum size supported by your GPU, or let `batch=-1` profile the model and target roughly 60% of GPU memory. If you encounter memory errors, incrementally reduce the batch size until training runs smoothly. Note that `batch=-1` only works on CUDA devices; on CPU and Apple silicon it falls back to `batch=16`. If utilization is unstable rather than capped, raise `workers` instead. For further information, refer to the [Train mode arguments](../modes/train.md).

### What is mixed precision training, and how do I enable it in YOLO26?

Mixed precision training utilizes both 16-bit (FP16) and 32-bit (FP32) floating-point types to balance computational speed and precision. This approach speeds up training and reduces memory usage without sacrificing model [accuracy](https://www.ultralytics.com/glossary/accuracy). In YOLO26 it is enabled by default through the `amp` parameter, so there is nothing to switch on; set `amp=False` to force FP32 instead. AMP is skipped on CPU and Apple silicon; CUDA, XPU, and NPU devices each have their own scaler path. For more details, see the [full list of training settings](../modes/train.md).

### How does multi-scale training enhance YOLO26 model performance?

Multi-scale training enhances model performance by training on images of varying sizes, allowing the model to better generalize across different scales and distances. YOLO26 offers two separate parameters for this. `scale=0.5` samples a zoom factor between 0.5 and 1.5 and then pads or crops back to a fixed `imgsz`, while `multi_scale=0.25` varies `imgsz` itself each batch in stride steps. For settings and more details, check out the [Train mode documentation](../modes/train.md).

### How can I use pretrained weights to speed up training in YOLO26?

Using pretrained weights can greatly accelerate training and enhance model accuracy by leveraging a model already familiar with foundational visual features. Load them through the `model` argument, as in `model=yolo26n.pt`; `pretrained` then decides whether those weights are kept (`True`, the default), discarded (`False`), or replaced by another checkpoint (`pretrained=path/to/best.pt`, the way to transfer weights into a `model=*.yaml` build). Learn more in the [Train mode documentation](../modes/train.md).

### Should I use `cache=True` or `cache='disk'` when training YOLO?

Use `cache=True` when the dataset fits in RAM — it gives the fastest data loading and keeps the GPU from idling between batches. Use `cache='disk'` when the dataset is larger than available memory; it is slower than RAM but still avoids re-decoding source images every epoch. Leave `cache=False` (the default) if memory and disk space are both tight, and prefer `cache='disk'` when a run has to be reproducible. See [Caching](#caching) for the full comparison.

### Which optimizer should I use for YOLO26 training?

Start with `optimizer=auto`, which picks AdamW for short runs and MuSGD for long ones based on the total iteration count. Override it only when training is unstable or when you need to control `lr0` and `momentum` yourself, since `optimizer=auto` ignores those values. See [Selecting an Optimizer](#selecting-an-optimizer) for the tradeoffs of each choice.

### How do I stop YOLO training early when the model stops improving?

Set the `patience` parameter to the number of epochs to wait for a validation improvement before stopping — `patience=5` ends training after 5 epochs with no gain. This saves compute and guards against [overfitting](https://www.ultralytics.com/glossary/overfitting) without capping `epochs` conservatively up front. To bound a run by wall-clock time instead, set `time` to a number of hours.

### What is the recommended number of epochs for training a model, and how do I set this in YOLO26?

The number of epochs refers to the complete passes through the training dataset during model training. A typical starting point is 300 epochs. If your model overfits early, you can reduce the number. Alternatively, if overfitting isn't observed, you might extend training to 600, 1200, or more epochs. To set this in YOLO26, use the `epochs` parameter in your training script. For additional advice on determining the ideal number of epochs, refer to this section on [number of epochs](#the-number-of-epochs-to-train-for).
