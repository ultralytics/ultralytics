---
---
# YOLOv6-3.0 VS YOLO11

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv6-3.0**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLO11**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 37.5 | 39.5 |
| s | 45.0 | 47.0 |
| m | 50.0 | 51.4 |
| l | 52.8 | 53.2 |
| x | N/A | 54.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv6-3.0**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLO11**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.17 | 1.55 |
| s | 2.66 | 2.63 |
| m | 5.28 | 5.27 |
| l | 8.95 | 6.84 |
| x | N/A | 12.49 |
