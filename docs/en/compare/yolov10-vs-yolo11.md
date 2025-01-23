---
---
# YOLOv10 VS YOLO11

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv10**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLO11**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 39.5 | 39.5 |
| s | 46.7 | 47.0 |
| m | 51.3 | 51.4 |
| b | 52.7 | N/A |
| l | 53.3 | 53.2 |
| x | 54.4 | 54.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv10**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLO11**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.56 | 1.55 |
| s | 2.66 | 2.63 |
| m | 5.48 | 5.27 |
| b | 6.54 | N/A |
| l | 8.33 | 6.84 |
| x | 12.2 | 12.49 |
