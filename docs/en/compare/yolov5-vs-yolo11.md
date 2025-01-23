---
---
# YOLOv5 VS YOLO11

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv5**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLO11**</span></center> |
|----|----------------------------------|------------------------------------|
| n | N/A | 39.5 |
| s | 37.4 | 47.0 |
| m | 45.4 | 51.4 |
| l | 49.0 | 53.2 |
| x | 50.7 | 54.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv5**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLO11**</span></center> |
|---------|-----------------------|-----------------------|
| n | N/A | 1.55 |
| s | 1.92 | 2.63 |
| m | 4.03 | 5.27 |
| l | 6.61 | 6.84 |
| x | 11.89 | 12.49 |
