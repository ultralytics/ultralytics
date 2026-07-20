---
comments: true
description: Learn the core computer vision techniques - object detection, segmentation, tracking, pose estimation, OBB, and more - with conceptual explanations, code examples, and links to Ultralytics implementations.
keywords: computer vision techniques, object detection, image segmentation, instance segmentation, semantic segmentation, object tracking, pose estimation, optical flow, depth estimation, oriented bounding box, OBB, deep learning computer vision, YOLO
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "TechArticle",
      "headline": "Computer Vision Techniques",
      "description": "A comprehensive guide to the core computer vision techniques including object detection, segmentation, tracking, pose estimation, and more.",
      "author": {"@type": "Organization", "name": "Ultralytics"},
      "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
      "url": "https://docs.ultralytics.com/computer-vision/techniques/",
      "about": {"@type": "Thing", "name": "Computer Vision Techniques"}
    },
    {
      "@type": "FAQPage",
      "mainEntity": [
        {
          "@type": "Question",
          "name": "What is the difference between object detection and image classification?",
          "acceptedAnswer": {"@type": "Answer", "text": "Image classification assigns a single label to an entire image. Object detection locates and labels multiple objects within the same image, outputting a bounding box and class label for each one. Use classification when the visual question is global; use detection when you need to know where specific objects are."}
        },
        {
          "@type": "Question",
          "name": "What is the difference between instance segmentation and semantic segmentation?",
          "acceptedAnswer": {"@type": "Answer", "text": "Semantic segmentation labels every pixel with a class but does not distinguish between individual instances - all people are labelled 'person'. Instance segmentation assigns a separate pixel mask to each detected instance, so two people in the same frame get two distinct masks."}
        },
        {
          "@type": "Question",
          "name": "When should I use object tracking instead of detection?",
          "acceptedAnswer": {"@type": "Answer", "text": "Use tracking when you need to follow the same object across multiple video frames with a consistent ID. Detection alone re-identifies objects independently per frame with no memory across frames. Tracking enables speed estimation, counting through a line, queue analysis, and any application that requires knowing an object's trajectory over time."}
        },
        {
          "@type": "Question",
          "name": "What is an oriented bounding box and when do I need it?",
          "acceptedAnswer": {"@type": "Answer", "text": "An oriented bounding box (OBB) adds a rotation angle to the standard four-coordinate box, allowing it to tightly fit objects at any angle. Use OBB when your objects appear at arbitrary orientations - particularly in aerial or satellite imagery where vehicles, ships, and structures are rarely axis-aligned."}
        },
        {
          "@type": "Question",
          "name": "Can I combine multiple computer vision techniques in one pipeline?",
          "acceptedAnswer": {"@type": "Answer", "text": "Yes. Many production pipelines chain techniques together. A common pattern is detection followed by tracking, with classification or pose estimation applied to each tracked region. Ultralytics YOLO supports detection, segmentation, tracking, pose estimation, and OBB - often runnable from the same model checkpoint with different task heads."}
        }
      ]
    }
  ]
}
</script>

# Computer Vision Techniques

Part of the [Computer Vision Hub](../index.md).

Computer vision techniques are the algorithmic building blocks that turn raw image data into structured understanding. Each technique addresses a specific type of visual problem - from locating objects in a frame to mapping every pixel to a class label. This page explains the core approaches conceptually, when to use each, and how they connect to Ultralytics implementations.

!!! tip "Prerequisites"

    - Python 3.8+ and `pip install ultralytics`
    - A webcam, image file, or video — most techniques work out of the box with a pretrained model
    - GPU recommended for real-time video; CPU is fine for image-by-image inference

## Quick reference

| Technique | Output | Real-time capable | Best for |
|---|---|---|---|
| [Image Classification](#image-classification) | Single class label | Yes | Scene-level recognition |
| [Object Detection](#object-detection) | Bounding boxes + labels | Yes | Multi-object localisation |
| [Instance Segmentation](#instance-segmentation) | Per-object pixel masks | Yes | Precise shape extraction |
| [Semantic Segmentation](#semantic-segmentation) | Dense pixel class map | Partial | Scene understanding |
| [Object Tracking](#object-tracking) | IDs across frames | Yes | Video analysis, counting |
| [Pose Estimation](#pose-estimation) | Keypoint coordinates | Yes | Human movement analysis |
| [OBB Detection](#oriented-bounding-box-detection-obb) | Rotated bounding boxes | Yes | Aerial and rotated objects |
| [Optical Flow](#optical-flow) | Per-pixel motion vectors | Partial | Motion without identity |
| [Depth Estimation](#depth-estimation-and-3d-vision) | Distance per pixel | Partial | Spatial reasoning |

---

## Image Classification

Image classification assigns a single label to an entire image - the simplest computer vision task. The model outputs a probability distribution over all possible classes and the highest-scoring class is the prediction.

Classification is appropriate when the visual question is global ("is this a defect or not?") rather than spatial ("where is the defect?"). It is also commonly used as the final layer in a pipeline - for example, classifying a detected region into severity categories after detection has already isolated it.

![Image classification examples showing predicted class labels on images](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/image-classification-examples.avif)

**Ultralytics implementation:** [Image Classification](../../tasks/classify.md)

**See it in action:** [Live Inference App](../projects/index.md#3-live-inference-app) · [Fine-Tune on Custom Dataset](../projects/index.md#25-fine-tune-on-custom-dataset)

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n-cls.pt")
        results = model("image.jpg")
        print(results[0].probs.top1)  # top class index
        ```

---

## Object Detection

Object detection answers the question: what is in this image, and where is it? The model outputs a bounding box and class label for each detected object in a single forward pass.

Modern detectors fall into two broad families:

- **Two-stage detectors** (e.g. Faster R-CNN) generate region proposals first, then classify each one. More accurate but slower.
- **Single-stage detectors** (e.g. YOLO) predict boxes and classes simultaneously. Significantly faster and suitable for real-time applications.

The YOLO architecture, now in its YOLO11 generation, is the dominant single-stage approach for production deployments. It uses anchor-free prediction with decoupled detection heads, removing the need for manual anchor configuration.

![Object detection examples showing bounding boxes and class labels on multiple objects](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/object-detection-examples.avif)

**Ultralytics implementation:** [Object Detection](../../tasks/detect.md)

**See it in action:** [Security Alarm System](../projects/index.md#1-security-alarm-system) · [Object Counting](../projects/index.md#2-object-counting) · [Parking Management](../projects/index.md#7-parking-management)

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        results = model("image.jpg")
        for box in results[0].boxes:
            print(box.xyxy, box.cls, box.conf)
        ```

    === "CLI"

        ```bash
        yolo detect predict model=yolo11n.pt source=image.jpg
        ```

---

## Instance Segmentation

Instance segmentation extends object detection by producing a precise pixel-level mask for each detected object, distinguishing between individual instances of the same class. If five people appear in a frame, instance segmentation produces five separate masks - one per person.

Mask prediction is handled by a lightweight mask head running in parallel with the detection head, adding minimal latency overhead over a base detector.

**When to use over bounding boxes:** When the exact shape or boundary of an object matters - surgical instrument tracking, precise crop measurement, or object extraction from backgrounds.

![Instance segmentation examples showing per-object pixel masks in different colours](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/instance-segmentation-examples.avif)

**Ultralytics implementation:** [Instance Segmentation](../../tasks/segment.md) · [Instance Segmentation with Tracking](../../guides/instance-segmentation-and-tracking.md) · [Object Cropping](../../guides/object-cropping.md)

**See it in action:** [Object Blurring and Anonymisation](../projects/index.md#4-object-blurring-and-anonymisation) · [Instance Segmentation with Tracking](../projects/index.md#15-instance-segmentation-with-tracking)

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n-seg.pt")
        results = model("image.jpg")
        for mask in results[0].masks:
            print(mask.xy)  # polygon coordinates
        ```

    === "CLI"

        ```bash
        yolo segment predict model=yolo11n-seg.pt source=image.jpg
        ```

---

## Semantic Segmentation

Semantic segmentation labels every pixel in an image with a class category, without distinguishing between individual instances. All cars are labelled "car", all road surface is labelled "road" - the output is a dense class map over the entire scene.

This is the technique of choice when the question is "what type of surface or region is this?" rather than "which specific object is this?" Common architectures use encoder-decoder structures (e.g. U-Net, DeepLab) that downsample for context then upsample for spatial precision.

**When to use:** Autonomous driving scene understanding, satellite land-use mapping, medical tissue classification.

![Semantic segmentation example showing a dense pixel class map across a full scene](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/semantic-segmentation-examples.avif)

**Ultralytics implementation:** [Semantic Segmentation](../../tasks/semantic.md)

---

## Object Tracking

Object tracking extends detection across time - assigning a consistent identity to each detected object across video frames. The tracker links detections frame-to-frame using appearance features, motion prediction (e.g. Kalman filtering), and spatial proximity.

Two dominant paradigms:

- **Tracking-by-detection** (e.g. ByteTrack, BoTSORT) - run a detector per frame, then associate detections using a separate algorithm. Modular and widely used.
- **Joint detection and tracking** - learn detection and re-identification simultaneously in a single model. Higher accuracy but more complex to train.

**Derived capabilities built on tracking:** [Speed estimation](../../guides/speed-estimation.md), [queue management](../../guides/queue-management.md), [zone-based counting](../../guides/trackzone.md), [heatmap generation](../../guides/heatmaps.md).

![Multi-object tracking examples showing bounding boxes with consistent track IDs across frames](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/multi-object-tracking-examples.avif)

**Ultralytics implementation:** [Object Tracking](../../modes/track.md)

**See it in action:** [Speed Estimation](../projects/index.md#11-speed-estimation) · [Heatmap Generation](../projects/index.md#12-heatmap-generation) · [Queue Management](../projects/index.md#10-queue-management) · [Region and Zone Counting](../projects/index.md#13-region-and-zone-counting)

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        results = model.track("video.mp4", persist=True)
        for result in results:
            print(result.boxes.id)  # track IDs
        ```

    === "CLI"

        ```bash
        yolo track model=yolo11n.pt source=video.mp4
        ```

---

## Pose Estimation

Pose estimation detects a set of keypoints - typically joints or anatomical landmarks - and their spatial relationships to map the posture of a person or object. A skeleton graph connects the keypoints into a structural model.

The output is a set of `(x, y, confidence)` coordinates per keypoint per detected instance. Top-down approaches (detect person first, then estimate pose) give higher accuracy; bottom-up approaches (detect all keypoints first, then group) are faster at scale.

**Common applications:** Workout form analysis, ergonomics monitoring, sports biomechanics, gesture control, rehabilitation tracking.

![Pose estimation examples showing skeleton keypoint overlays on people with connected joints](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/pose-estimation-examples.avif)

**Ultralytics implementation:** [Pose Estimation](../../tasks/pose.md) · [Workouts Monitoring](../../guides/workouts-monitoring.md)

**See it in action:** [Workout and Rep Counter](../projects/index.md#8-workout-and-rep-counter) · [Pose Estimation and Ergonomics](../projects/index.md#18-pose-estimation-and-ergonomics)

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n-pose.pt")
        results = model("image.jpg")
        for keypoints in results[0].keypoints:
            print(keypoints.xy)  # (x, y) per keypoint
        ```

    === "CLI"

        ```bash
        yolo pose predict model=yolo11n-pose.pt source=image.jpg
        ```

---

## Oriented Bounding Box Detection (OBB)

Standard bounding boxes are axis-aligned rectangles - they cannot capture the true orientation of elongated or rotated objects. Oriented bounding box detection adds a rotation angle as a fifth parameter, allowing the box to tightly fit the object regardless of its orientation in the frame.

This is critical for aerial and satellite imagery, where vehicles, ships, and structures appear at arbitrary angles, and axis-aligned boxes would introduce significant background noise into the detection region.

**When to use:** Aerial imagery analysis, satellite object detection, document layout analysis, warehouse robotics with rotated goods.

![Oriented bounding box detection on aerial imagery showing tight-fitting rotated boxes on ships](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ships-detection-using-obb.avif)

**Ultralytics implementation:** [OBB Detection](../../tasks/obb.md)

**See it in action:** [Rotated Object Detection](../projects/index.md#19-rotated-object-detection-obb) · [Tiled Inference for Large Images](../projects/index.md#20-tiled-inference-for-large-images)

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n-obb.pt")
        results = model("aerial.jpg")
        for box in results[0].obb:
            print(box.xyxyxyxy)  # rotated box corners
        ```

    === "CLI"

        ```bash
        yolo obb predict model=yolo11n-obb.pt source=aerial.jpg
        ```

---

## Optical Flow

Optical flow estimates the apparent motion of pixels between consecutive frames - producing a dense vector field where each pixel has a velocity direction and magnitude. It does not require explicit object detection; instead it captures motion at the pixel level.

Applications include camera stabilisation, gesture recognition from motion rather than pose, and sports performance analysis where trajectory matters more than object identity.

![Speed and motion estimation on a bridge using Ultralytics YOLO](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/speed-estimation-on-bridge-using-ultralytics-yolov8.avif)

---

## Depth Estimation and 3D Vision

Monocular depth estimation infers the relative distance of each pixel from a single camera, using learned scene priors. Stereo depth estimation uses two calibrated cameras to compute disparity and recover true metric depth. LiDAR adds precise 3D point cloud data that computer vision models can fuse with camera inputs.

![3D point cloud segmentation showing depth and spatial structure of a scene](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/point-cloud-segmentation-ultralytics.avif)

[Distance calculation](../../guides/distance-calculation.md) between detected objects is a practical implementation of spatial reasoning built on top of detection output.

---

## Choosing the right technique

| Visual question | Technique |
|---|---|
| What class is this entire image? | [Image Classification](../../tasks/classify.md) |
| Where are specific objects, with bounding boxes? | [Object Detection](../../tasks/detect.md) |
| Where are objects, with pixel-precise outlines? | [Instance Segmentation](../../tasks/segment.md) |
| What class does every pixel belong to? | [Semantic Segmentation](../../tasks/semantic.md) |
| Where are objects across video frames, with identity? | [Object Tracking](../../modes/track.md) |
| What is the posture or joint configuration of a subject? | [Pose Estimation](../../tasks/pose.md) |
| Where are rotated or angled objects in aerial imagery? | [OBB Detection](../../tasks/obb.md) |

For a full end-to-end walkthrough of how these techniques fit into a real project, see the [steps of a CV project guide](../../guides/steps-of-a-cv-project.md).

Ready to apply these techniques? The [Computer Vision Projects](../projects/index.md) library has 25+ hands-on examples, each with working code and no custom training required to start.

---

## FAQ

??? question "What is the difference between object detection and image classification?"

    Image classification assigns a single label to an entire image. Object detection locates and labels multiple objects within the same image, outputting a bounding box and class label for each one. Use classification when the visual question is global; use detection when you need to know where specific objects are in the frame.

??? question "What is the difference between instance segmentation and semantic segmentation?"

    Semantic segmentation labels every pixel with a class but does not distinguish between individual instances - all people are labelled "person". Instance segmentation assigns a separate pixel mask to each detected instance, so two people in the same frame get two distinct masks. Instance segmentation is more computationally expensive but far more useful when objects of the same class need to be treated separately.

??? question "When should I use object tracking instead of detection?"

    Use tracking when you need to follow the same object across multiple video frames with a consistent ID. Detection alone re-identifies objects independently per frame with no memory across frames. Tracking enables speed estimation, counting through a line, queue analysis, and any application that requires knowing an object's trajectory over time.

??? question "What is an oriented bounding box and when do I need it?"

    An oriented bounding box (OBB) adds a rotation angle to the standard four-coordinate box, allowing it to tightly fit objects at any angle. Use OBB when your objects appear at arbitrary orientations - particularly in aerial or satellite imagery where vehicles, ships, and structures are rarely axis-aligned.

??? question "Can I combine multiple computer vision techniques in one pipeline?"

    Yes. Many production pipelines chain techniques together. A common pattern is detection followed by tracking, with classification or pose estimation applied to each tracked region. Ultralytics YOLO supports detection, segmentation, tracking, pose estimation, and OBB - often runnable from the same model checkpoint with different task heads.

---

## Community and support

Have a question about which technique fits your project, or want to share what you've built?

- [GitHub Discussions](https://github.com/ultralytics/ultralytics/discussions) — ask questions and get help from the team
- [Discord](https://discord.com/invite/ultralytics) — real-time chat with other developers and Ultralytics engineers
- [Ultralytics Community Forum](https://community.ultralytics.com/) — longer-form discussion and project showcases
- [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) — report bugs or request features
