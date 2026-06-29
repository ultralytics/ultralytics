---
comments: true
description: Explore the core computer vision techniques — object detection, segmentation, tracking, pose estimation, and more — with conceptual explanations and links to Ultralytics implementations.
keywords: computer vision techniques, object detection, image segmentation, object tracking, pose estimation, optical flow, depth estimation, deep learning, YOLO
---

# Computer Vision Techniques

Computer vision techniques are the algorithmic building blocks that turn raw image data into structured understanding. Each technique addresses a specific type of visual problem — from locating objects in a frame to mapping every pixel to a class label. This page explains the core approaches conceptually, when to use each, and how they connect to Ultralytics implementations.

## Object Detection

Object detection answers the question: **what is in this image, and where is it?** The model outputs a bounding box and class label for each detected object in a single forward pass.

Modern detectors fall into two broad families:

- **Two-stage detectors** (e.g. Faster R-CNN) generate region proposals first, then classify each one. More accurate but slower.
- **Single-stage detectors** (e.g. YOLO) predict boxes and classes simultaneously across a grid. Significantly faster — suitable for real-time applications.

The YOLO architecture, now in its YOLO26 generation, is the dominant single-stage approach for production deployments. It uses anchor-free prediction with decoupled detection heads, removing the need for manual anchor configuration.

**Ultralytics implementation:** [Object Detection](../../tasks/detect.md)

---

## Instance Segmentation

Instance segmentation extends object detection by producing a precise **pixel-level mask** for each detected object, distinguishing between individual instances of the same class. Unlike semantic segmentation, if five people appear in a frame, instance segmentation produces five separate masks — one per person.

Mask prediction is typically handled by a lightweight mask head running in parallel with the detection head, adding minimal latency overhead over a base detector.

**When to use over bounding boxes:** When the exact shape or boundary of an object matters — surgical instrument tracking, precise crop measurement, object extraction from backgrounds.

**Ultralytics implementation:** [Instance Segmentation](../../tasks/segment.md) · [Instance Segmentation with Tracking](../../guides/instance-segmentation-and-tracking.md) · [Object Cropping](../../guides/object-cropping.md)

---

## Semantic Segmentation

Semantic segmentation labels **every pixel** in an image with a class category, without distinguishing between individual instances. All cars are labelled "car", all road surface is labelled "road" — the output is a dense class map over the entire scene.

This is the technique of choice when the question is "what type of surface or region is this?" rather than "which specific object is this?" Common architectures use encoder-decoder structures (e.g. U-Net, DeepLab) that downsample for context then upsample for spatial precision.

**When to use:** Autonomous driving scene understanding, satellite land-use mapping, medical tissue classification.

**Ultralytics implementation:** [Semantic Segmentation](../../tasks/semantic.md)

---

## Object Tracking

Object tracking extends detection across time — assigning a consistent identity to each detected object across video frames. The tracker links detections frame-to-frame using appearance features, motion prediction (e.g. Kalman filtering), and spatial proximity.

Two dominant paradigms:

- **Tracking-by-detection** (e.g. ByteTrack, BoTSORT) — run a detector per frame, then associate detections using a separate algorithm. Modular and widely used.
- **Joint detection and tracking** — learn detection and re-identification simultaneously in a single model. Higher accuracy but more complex to train.

**Derived capabilities built on tracking:** [Speed estimation](../../guides/speed-estimation.md), [queue management](../../guides/queue-management.md), [zone-based counting](../../guides/trackzone.md), [heatmap generation](../../guides/heatmaps.md).

**Ultralytics implementation:** [Object Tracking](../../modes/track.md) · [Tracking Practical Guide](../../guides/tracking-practical-guide.md)

---

## Pose Estimation

Pose estimation detects a set of **keypoints** — typically joints or anatomical landmarks — and their spatial relationships to map the posture of a person or object. A skeleton graph connects the keypoints into a structural model.

The output is a set of `(x, y, confidence)` coordinates per keypoint per detected instance. Top-down approaches (detect person first, then estimate pose) give higher accuracy; bottom-up approaches (detect all keypoints first, then group) are faster at scale.

**Common applications:** Workout form analysis, ergonomics monitoring, sports biomechanics, gesture control, rehabilitation tracking.

**Ultralytics implementation:** [Pose Estimation](../../tasks/pose.md) · [Pose Practical Guide](../../guides/pose-practical-guide.md) · [Workouts Monitoring](../../guides/workouts-monitoring.md)

---

## Oriented Bounding Box Detection (OBB)

Standard bounding boxes are axis-aligned rectangles — they cannot capture the true orientation of elongated or rotated objects. Oriented bounding box detection adds a **rotation angle** as a fifth parameter, allowing the box to tightly fit the object regardless of its orientation in the frame.

This is critical for aerial and satellite imagery, where vehicles, ships, and structures appear at arbitrary angles, and axis-aligned boxes would introduce significant background noise into the detection region.

**When to use:** Aerial imagery analysis, satellite object detection, document layout analysis, warehouse robotics with rotated goods.

**Ultralytics implementation:** [OBB Detection](../../tasks/obb.md)

---

## Image Classification

Image classification assigns a **single label** to an entire image — the simplest computer vision task. The model outputs a probability distribution over all possible classes, and the highest-scoring class is the prediction.

Classification is appropriate when the visual question is global ("is this a cat or a dog?") rather than spatial ("where is the cat?"). It is also commonly used as the final layer of a pipeline — for example, classifying a detected defect into severity categories after detection has already isolated the region.

**Ultralytics implementation:** [Image Classification](../../tasks/classify.md)

---

## Optical Flow

Optical flow estimates the **apparent motion** of pixels between consecutive frames — producing a dense vector field where each pixel has a velocity direction and magnitude. It does not require explicit object detection; instead it captures motion at the pixel level.

Applications include drone stabilisation, gesture recognition from motion rather than pose, and sports performance analysis where trajectory matters more than identity.

---

## Depth Estimation and 3D Vision

Monocular depth estimation infers the relative distance of each pixel from a single camera, using learned scene priors. Stereo depth estimation uses two calibrated cameras to compute disparity and recover true metric depth. LiDAR adds precise 3D point cloud data that computer vision models can fuse with camera inputs.

[Distance calculation](../../guides/distance-calculation.md) between detected objects is a practical implementation of spatial reasoning built on top of detection output.

---

## Choosing the Right Technique

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
