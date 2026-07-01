---
comments: true
description: Explore real-world computer vision applications across retail, healthcare, manufacturing, agriculture, sports, security, and autonomous vehicles - with working code examples and links to Ultralytics implementations.
keywords: computer vision applications, computer vision in retail, computer vision in healthcare, computer vision in manufacturing, computer vision in agriculture, computer vision in sports, computer vision security, autonomous vehicles computer vision, YOLO applications
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "TechArticle",
      "headline": "Computer Vision Applications",
      "description": "Real-world applications of computer vision across industries including retail, healthcare, manufacturing, agriculture, sports, security, and autonomous vehicles.",
      "author": {"@type": "Organization", "name": "Ultralytics"},
      "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
      "url": "https://docs.ultralytics.com/computer-vision/applications/",
      "about": {"@type": "Thing", "name": "Computer Vision Applications"}
    },
    {
      "@type": "FAQPage",
      "mainEntity": [
        {
          "@type": "Question",
          "name": "What industries use computer vision most?",
          "acceptedAnswer": {"@type": "Answer", "text": "Computer vision is deployed across virtually every industry. The highest-adoption sectors are retail (loss prevention, inventory, checkout automation), manufacturing (defect detection, quality control), healthcare (medical imaging, surgical assistance), automotive (ADAS, autonomous driving), agriculture (crop monitoring, yield estimation), security (surveillance, access control), and sports (player tracking, performance analytics)."}
        },
        {
          "@type": "Question",
          "name": "Do I need custom training for every application?",
          "acceptedAnswer": {"@type": "Answer", "text": "Not always. Pretrained YOLO models work immediately for common objects (people, vehicles, animals, everyday items) without any training. Custom training is needed when your target objects are not in the pretrained class set - for example, specific product SKUs, medical devices, or industrial components. Many of the applications on this page can be prototyped with a pretrained model before committing to custom training."}
        },
        {
          "@type": "Question",
          "name": "Can computer vision run in real time on edge devices?",
          "acceptedAnswer": {"@type": "Answer", "text": "Yes. YOLO11n (nano) runs at over 100 FPS on a modern CPU and significantly faster on a GPU or dedicated edge accelerator. Ultralytics supports export to ONNX, TensorRT, CoreML, and other formats optimised for edge deployment on NVIDIA Jetson, Raspberry Pi, and similar hardware."}
        },
        {
          "@type": "Question",
          "name": "How accurate is computer vision for industrial defect detection?",
          "acceptedAnswer": {"@type": "Answer", "text": "Accuracy depends heavily on training data quality and the complexity of defect types. Custom-trained YOLO models regularly achieve over 95% precision on well-defined defect categories when trained on representative datasets. Tiled inference is often used for high-resolution inspection images to ensure small defects are not missed."}
        }
      ]
    }
  ]
}
</script>

# Computer Vision Applications

Part of the [Computer Vision Hub](../index.md).

Computer vision has moved from research labs into production across nearly every industry. This page covers the most impactful real-world application areas, how they work technically, and how to implement them using Ultralytics YOLO.

## Retail

Computer vision gives retailers real-time visibility into what is happening on the shop floor - without manual auditing.

**Core use cases:**

- **Loss prevention** - detect theft behaviours, monitor checkout lanes, and flag unscanned items
- **Inventory management** - track shelf stock levels, identify out-of-stock positions, and automate replenishment alerts
- **Customer analytics** - measure footfall, dwell time, queue lengths, and conversion zones
- **Cashierless checkout** - identify products as customers pick them up and charge automatically

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")

        # Count people in a store zone
        results = model.track("store_feed.mp4", persist=True, classes=[0])  # class 0 = person
        ```

    === "CLI"

        ```bash
        yolo track model=yolo11n.pt source=store_feed.mp4 classes=0
        ```

**Relevant guides:** [Object Counting](../../guides/object-counting.md) · [Queue Management](../../guides/queue-management.md) · [Heatmaps](../../guides/heatmaps.md) · [Region Counting](../../guides/region-counting.md)

---

## Healthcare and Medical Imaging

Computer vision assists clinicians by automating the analysis of medical images at a scale and consistency that manual review cannot match.

**Core use cases:**

- **Radiology** - detect anomalies in X-ray, CT, and MRI scans to flag cases for review
- **Pathology** - classify tissue samples and identify cellular abnormalities in histology slides
- **Surgical assistance** - track instruments, anatomical landmarks, and tissue boundaries during procedures
- **Patient monitoring** - detect falls, measure vital signs from video, or monitor post-surgical recovery remotely

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Fine-tuned model on medical imagery
        model = YOLO("yolo11n-seg.pt")
        results = model("scan.jpg")

        for mask in results[0].masks:
            print(mask.xy)  # segmentation boundary coordinates
        ```

**Relevant guides:** [Instance Segmentation](../../tasks/segment.md) · [Fine-Tune on Custom Dataset](../../modes/train.md)

---

## Manufacturing and Quality Control

Automated visual inspection replaces or augments manual quality checks on production lines, catching defects faster and more consistently than human inspectors.

**Core use cases:**

- **Surface defect detection** - identify scratches, dents, discolouration, and cracks on manufactured parts
- **Assembly verification** - confirm that all components are present and correctly positioned
- **Dimensional measurement** - use pixel dimensions with known camera calibration to verify tolerances
- **Packaging inspection** - check label placement, seal integrity, and fill levels

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")  # or a custom-trained defect model

        # Tiled inference for high-resolution inspection images
        results = model("part.jpg", imgsz=1280)
        for det in results[0].boxes:
            print(f"Defect: {det.cls}, confidence: {det.conf:.2f}")
        ```

    === "CLI"

        ```bash
        yolo detect predict model=best.pt source=part.jpg imgsz=1280
        ```

**Relevant guides:** [Tiled Inference for Large Images](../projects/index.md#20-tiled-inference-for-large-images) · [Fine-Tune on Custom Dataset](../../modes/train.md) · [OBB Detection](../../tasks/obb.md)

---

## Agriculture

Computer vision enables precision agriculture by providing detailed, timely data about crops, livestock, and field conditions that would be impractical to collect manually.

**Core use cases:**

- **Crop monitoring** - identify disease, pest damage, or nutrient deficiency from drone or ground imagery
- **Yield estimation** - count fruit or grain heads to predict harvest quantities before picking
- **Weed detection** - distinguish crop plants from weeds for targeted herbicide application
- **Livestock monitoring** - track animal behaviour, detect injury or illness, and count herd size

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")

        # Count fruit on a tree from drone imagery
        results = model("orchard.jpg", imgsz=1280)
        fruit_count = len(results[0].boxes)
        print(f"Estimated yield: {fruit_count} items visible")
        ```

    === "CLI"

        ```bash
        yolo detect predict model=yolo11n.pt source=orchard.jpg imgsz=1280
        ```

**Relevant guides:** [Object Counting](../../guides/object-counting.md) · [Tiled Inference for Large Images](../projects/index.md#20-tiled-inference-for-large-images)

---

## Sports and Performance Analysis

Computer vision extracts objective performance data from broadcast footage and training video, giving coaches and analysts insights that manual observation cannot produce at scale.

**Core use cases:**

- **Player tracking** - follow individual players across a full pitch or court, measuring distance covered and speed
- **Ball and equipment tracking** - analyse trajectory, spin, and impact points
- **Pose and biomechanics** - measure joint angles, stride length, and movement patterns for technique coaching
- **Event detection** - automatically flag goals, fouls, or specific plays for highlight generation

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n-pose.pt")
        results = model.track("training_clip.mp4", persist=True)

        for result in results:
            # Keypoints per detected athlete
            if result.keypoints:
                print(result.keypoints.xy)
        ```

    === "CLI"

        ```bash
        yolo pose track model=yolo11n-pose.pt source=training_clip.mp4
        ```

**Relevant guides:** [Pose Estimation](../../tasks/pose.md) · [Speed Estimation](../../guides/speed-estimation.md) · [Workouts Monitoring](../../guides/workouts-monitoring.md) · [Object Tracking](../../modes/track.md)

---

## Security and Surveillance

Computer vision makes security systems proactive rather than reactive - detecting events as they happen rather than reviewing footage after the fact.

**Core use cases:**

- **Intrusion detection** - alert when a person enters a restricted zone
- **Perimeter monitoring** - track movement along boundaries and flag loitering
- **Access control** - verify identity through face or body recognition at entry points
- **Crowd management** - measure crowd density and flow to prevent overcrowding incidents

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO
        from ultralytics import solutions

        # Security alarm that triggers on new object detection
        alarm = solutions.SecurityAlarm(
            model="yolo11n.pt",
            show=True,
        )

        import cv2
        cap = cv2.VideoCapture("feed.mp4")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            alarm(frame)
        ```

    === "CLI"

        ```bash
        yolo solutions security model=yolo11n.pt source=feed.mp4
        ```

**Relevant guides:** [Security Alarm System](../../guides/security-alarm-system.md) · [Zone Tracking](../../guides/trackzone.md) · [Region Counting](../../guides/region-counting.md)

---

## Autonomous Vehicles and ADAS

Computer vision is a core sensing modality for any system that navigates or operates in the physical world without direct human control.

**Core use cases:**

- **Pedestrian and vehicle detection** - identify road users in real time for collision avoidance
- **Lane detection and tracking** - understand road structure for lane-keeping assistance
- **Traffic sign recognition** - read and respond to signs, signals, and road markings
- **Depth and obstacle estimation** - build a spatial model of the environment from camera feeds

!!! example "Using Ultralytics YOLO"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")

        # Run on a dashcam feed
        results = model("dashcam.mp4", stream=True)
        for result in results:
            # Filter for vehicles and pedestrians
            relevant = [b for b in result.boxes if int(b.cls) in [0, 2, 5, 7]]
            print(f"Detected {len(relevant)} road users")
        ```

    === "CLI"

        ```bash
        yolo detect predict model=yolo11n.pt source=dashcam.mp4 classes=0,2,5,7
        ```

**Relevant guides:** [Object Detection](../../tasks/detect.md) · [Distance Calculation](../../guides/distance-calculation.md) · [Speed Estimation](../../guides/speed-estimation.md) · [OBB Detection](../../tasks/obb.md)

---

## Choosing where to start

| If you want to... | Start here |
|---|---|
| Prototype quickly with no training | [Computer Vision Projects](../projects/index.md) - 25+ ready-to-run examples |
| Understand the underlying techniques | [Computer Vision Techniques](../techniques/index.md) |
| Build for a specific industry | The relevant section above |
| Train on your own data | [Train a custom model](../../modes/train.md) |
| Deploy to edge hardware | [NVIDIA Jetson](../../guides/nvidia-jetson.md) · [Raspberry Pi](../../guides/raspberry-pi.md) |

---

## FAQ

??? question "What industries use computer vision most?"

    Computer vision is deployed across virtually every industry. The highest-adoption sectors are retail (loss prevention, inventory, checkout automation), manufacturing (defect detection, quality control), healthcare (medical imaging, surgical assistance), automotive (ADAS, autonomous driving), agriculture (crop monitoring, yield estimation), security (surveillance, access control), and sports (player tracking, performance analytics).

??? question "Do I need custom training for every application?"

    Not always. Pretrained YOLO models work immediately for common objects - people, vehicles, animals, and everyday items - without any training. Custom training is needed when your target objects are not in the pretrained class set, for example specific product SKUs, medical devices, or industrial components. Many applications can be prototyped with a pretrained model before committing to custom training.

??? question "Can computer vision run in real time on edge devices?"

    Yes. YOLO11n (nano) runs at over 100 FPS on a modern CPU and significantly faster on a GPU or dedicated edge accelerator. Ultralytics supports export to ONNX, TensorRT, CoreML, and other formats optimised for edge deployment on NVIDIA Jetson, Raspberry Pi, and similar hardware.

??? question "How accurate is computer vision for industrial defect detection?"

    Accuracy depends heavily on training data quality and the complexity of defect types. Custom-trained YOLO models regularly achieve over 95% precision on well-defined defect categories when trained on representative datasets. Tiled inference is often used for high-resolution inspection images to ensure small defects are not missed.
