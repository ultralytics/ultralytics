---
comments: true
description: Analytics dashboard for monitoring the detection, tracking and other data. Retail analytics, traffic analytics
keywords: Ultralytics, dashboard, object detection, YOLO, YOLO model training, object tracking, computer vision, deep learning models, end user analytics
---

# Ultralytics Analytics Dashboard

Ultralytics Analytics offers a comprehensive dashboard that enables users to visualize analytics derived from inference data. This includes tracking information, object counting, line graphs, bar plots, and various other visuals, facilitating a clear and detailed comprehension of the data.

|                                               Ultralytics Analytics Dashboard (Beta v1.0)                                                |                                                                
|:----------------------------------------------------------------------------------------------------------------------------------------:|
| ![Ultralytics Analytics Dashboard ](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/ea8ed16e-5426-4b7e-9b3d-1913e2f66051) |


## Dashboard Configuration

Ultralytics provides a user-friendly configuration interface for the dashboard, allowing users to customize it for their specific needs. This enables users to run the dashboard with custom videos, modify regions, and perform multiple tasks without the need to directly alter any code.

!!! Tip "Tip"

    We currently support only specific arguments in dashboard (Beta v1.0) that are mentioned below, we are working on addition of multiple argument in coming releases. You can modify only the mentioned arguments at this point.


!!! Example "ultralytics/cfg/analytics/dashboard.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/analytics/dashboard.yaml"
    ```


## Custom Configuration File (template.yaml)

```yaml
mode: "mode name"  # supported modes predict, count, track
model: "model file path" # supported models yolov8s, yolov8m, yolov8l
video_path: "video file path"  # path to video file

# adjust according to your needs only supported for object counting
region_points: "[(20, 400), (1080, 404), (1080, 360), (20, 360)]"
```

## Run Analytics Dashboard

!!! Dashboard
    
    === "Python"
        ```python
        from ultralytics.cfg import handle_dashboard
        
        # Build a dashboard with ultralytics demo video
        handle_dashboard()

        # Build dashboard with custom configuration
        handle_dashboard("path/to/template.yaml")
        ```

    === "CLI"
        ```bash
        # Build a dashboard with ultralytics demo video
        yolo dashboard
        
        # Build dashboard with custom configuration
        yolo dashboard "path/to/template.yaml"
        ```

## Citations and Acknowledgments

The dashboard is officially supported by [Ultralytics](https://ultralytics.com/) and has been released available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
