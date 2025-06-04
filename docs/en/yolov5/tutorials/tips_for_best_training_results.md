---
comments: true
description: Discover how to achieve optimal mAP and training results using YOLOv5. Learn essential dataset, model selection, and training settings best practices.
keywords: YOLOv5 training, mAP, dataset best practices, model selection, training settings, YOLOv5 guide, YOLOv5 tutorial, machine learning
---

# Tips for Best YOLOv5 Training Results

📚 This guide explains how to produce the best mAP and training results with YOLOv5 🚀.

Most of the time good results can be obtained with no changes to the models or training settings, **provided your dataset is sufficiently large and well labelled**. If at first you don't get good results, there are steps you might be able to take to improve, but we always recommend users **first train with all default settings** before considering any changes. This helps establish a performance baseline and spot areas for improvement.

If you have questions about your training results **we recommend you provide the maximum amount of information possible** if you expect a helpful response, including results plots (train losses, val losses, P, R, mAP), PR curve, [confusion matrix](https://www.ultralytics.com/glossary/confusion-matrix), training mosaics, test results and dataset statistics images such as labels.png. All of these are located in your `project/name` directory, typically `yolov5/runs/train/exp`.

We've put together a full guide for users looking to get the best results on their YOLOv5 trainings below.

## Dataset

- **Images per class.** ≥ 1500 images per class recommended
- **Instances per class.** ≥ 10000 instances (labeled objects) per class recommended
- **Image variety.** Must be representative of deployed environment. For real-world use cases we recommend images from different times of day, different seasons, different weather, different lighting, different angles, different sources (scraped online, collected locally, different cameras) etc.
- **Label consistency.** All instances of all classes in all images must be labelled. Partial labelling will not work.
- **Label [accuracy](https://www.ultralytics.com/glossary/accuracy).** Labels must closely enclose each object. No space should exist between an object, and it's [bounding box](https://www.ultralytics.com/glossary/bounding-box). No objects should be missing a label.
- **Label verification.** View `train_batch*.jpg` on train start to verify your labels appear correct, i.e. see [example](./train_custom_data.md#local-logging) mosaic.
- **Background images.** Background images are images with no objects that are added to a dataset to reduce False Positives (FP). We recommend about 0-10% background images to help reduce FPs (COCO has 1000 background images for reference, 1% of the total). No labels are required for background images.

<a href="https://arxiv.org/abs/1405.0312"><img width="800" src="https://github.com/ultralytics/docs/releases/download/0/coco-analysis.avif" alt="COCO Analysis"></a>

## Model Selection

Larger models like YOLOv5x and [YOLOv5x6](https://github.com/ultralytics/yolov5/releases/tag/v5.0) will produce better results in nearly all cases, but have more parameters, require more CUDA memory to train, and are slower to run. For **mobile** deployments we recommend YOLOv5s/m, for **cloud** deployments we recommend YOLOv5l/x. See our README [table](https://github.com/ultralytics/yolov5#pretrained-checkpoints) for a full comparison of all models.

<p align="center"><img width="700" alt="YOLOv5 Models" src="https://github.com/ultralytics/docs/releases/download/0/yolov5-model-comparison.avif"></p>

- **Start from Pretrained weights.** Recommended for small to medium-sized datasets (i.e. [VOC](https://github.com/ultralytics/yolov5/blob/master/data/VOC.yaml), [VisDrone](https://github.com/ultralytics/yolov5/blob/master/data/VisDrone.yaml), [GlobalWheat](https://github.com/ultralytics/yolov5/blob/master/data/GlobalWheat2020.yaml)). Pass the name of the model to the `--weights` argument. Models download automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases).

    ```bash
    python train.py --data custom.yaml --weights yolov5s.pt
    python train.py --data custom.yaml --weights yolov5m.pt
    python train.py --data custom.yaml --weights yolov5l.pt
    python train.py --data custom.yaml --weights yolov5x.pt
    python train.py --data custom.yaml --weights custom_pretrained.pt
    ```

- **Start from Scratch.** Recommended for large datasets (i.e. [COCO](https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml), [Objects365](https://github.com/ultralytics/yolov5/blob/master/data/Objects365.yaml), [OIv6](https://storage.googleapis.com/openimages/web/index.html)). Pass the model architecture YAML you are interested in, along with an empty `--weights ''` argument:

    ```bash
    python train.py --data custom.yaml --weights '' --cfg yolov5s.yaml
    python train.py --data custom.yaml --weights '' --cfg yolov5m.yaml
    python train.py --data custom.yaml --weights '' --cfg yolov5l.yaml
    python train.py --data custom.yaml --weights '' --cfg yolov5x.yaml
    ```

## Training Settings

Before modifying anything, **first train with default settings to establish a performance baseline**. A full list of train.py settings can be found in the [train.py](https://github.com/ultralytics/yolov5/blob/master/train.py) argparser.

- **[Epochs](https://www.ultralytics.com/glossary/epoch).** Start with 300 epochs. If this overfits early then you can reduce epochs. If [overfitting](https://www.ultralytics.com/glossary/overfitting) does not occur after 300 epochs, train longer, i.e. 600, 1200 etc. epochs.
- **Image size.** COCO trains at native resolution of `--img 640`, though due to the high amount of small objects in the dataset it can benefit from training at higher resolutions such as `--img 1280`. If there are many small objects then custom datasets will benefit from training at native or higher resolution. Best inference results are obtained at the same `--img` as the training was run at, i.e. if you train at `--img 1280` you should also test and detect at `--img 1280`.
- **[Batch size](https://www.ultralytics.com/glossary/batch-size).** Use the largest `--batch-size` that your hardware allows for. Small batch sizes produce poor [batch normalization](https://www.ultralytics.com/glossary/batch-normalization) statistics and should be avoided. You can use `--batch-size -1` to automatically select the optimal batch size for your GPU.
- **[Learning rate](https://www.ultralytics.com/glossary/learning-rate).** The default learning rate schedule works well in most cases. For faster convergence, you can try using the `--cos-lr` flag to enable cosine learning rate scheduling, which gradually reduces the learning rate following a cosine curve over epochs.
- **[Data augmentation](https://www.ultralytics.com/glossary/data-augmentation).** YOLOv5 includes various augmentation techniques like mosaic, which combines multiple training images. For the last few epochs, consider using `--close-mosaic 10` to disable mosaic augmentation, which can help stabilize training.
- **Hyperparameters.** Default hyperparameters are in [hyp.scratch-low.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml). We recommend you train with default hyperparameters first before thinking of modifying any. In general, increasing augmentation hyperparameters will reduce and delay overfitting, allowing for longer trainings and higher final mAP. Reduction in loss component gain hyperparameters like `hyp['obj']` will help reduce overfitting in those specific loss components. For an automated method of optimizing these hyperparameters, see our [Hyperparameter Evolution Tutorial](./hyperparameter_evolution.md).
- **[Mixed precision](https://www.ultralytics.com/glossary/mixed-precision) training.** Enable mixed precision training with `--amp` to speed up training and reduce memory usage without sacrificing model accuracy.
- **Multi-GPU training.** If you have multiple GPUs, use `--device 0,1,2,3` to distribute training across them, which can significantly reduce training time.
- **Early stopping.** Use `--patience 50` to stop training if validation metrics don't improve for 50 epochs, saving time and preventing overfitting.

## Advanced Optimization Techniques

- **[Transfer learning](https://www.ultralytics.com/glossary/transfer-learning).** For specialized datasets, start with pretrained weights and gradually unfreeze layers during training to adapt the model to your specific task.
- **[Model pruning](https://www.ultralytics.com/glossary/model-pruning).** After training, consider pruning your model to remove redundant weights and reduce model size without significant performance loss.
- **[Model ensemble](https://www.ultralytics.com/glossary/model-ensemble).** For critical applications, train multiple models with different configurations and combine their predictions for improved accuracy.
- **[Test-time augmentation](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/).** Enable TTA during inference with `--augment` to improve prediction accuracy by averaging results from augmented versions of the input image.

## Further Reading

If you'd like to know more, a good place to start is Karpathy's 'Recipe for Training [Neural Networks](https://www.ultralytics.com/glossary/neural-network-nn)', which has great ideas for training that apply broadly across all ML domains: [https://karpathy.github.io/2019/04/25/recipe/](https://karpathy.github.io/2019/04/25/recipe/)

For more detailed information on training settings and configurations, refer to the [Ultralytics train settings documentation](https://docs.ultralytics.com/modes/train/), which provides comprehensive explanations of all available parameters.

Good luck 🍀 and let us know if you have any other questions!

## FAQ

### How do I know if my model is overfitting?

Your model may be overfitting if the training loss continues to decrease while validation loss starts to increase. Monitor the validation mAP - if it plateaus or decreases while training loss keeps improving, that's a sign of overfitting. Solutions include adding more training data, increasing data augmentation, or implementing regularization techniques.

### What's the optimal batch size for training YOLOv5?

The optimal batch size depends on your GPU memory. Larger batch sizes generally provide better batch normalization statistics and training stability. Use the largest batch size your hardware can handle without running out of memory. You can use `--batch-size -1` to automatically determine the optimal batch size for your setup.

### How can I speed up YOLOv5 training?

To speed up training, try: enabling mixed precision training with `--amp`, using multiple GPUs with `--device 0,1,2,3`, caching your dataset with `--cache`, and optimizing your batch size. Also consider using a smaller model variant like YOLOv5s if absolute accuracy isn't critical.
