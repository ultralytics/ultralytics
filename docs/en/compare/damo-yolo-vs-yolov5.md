---
---
# DAMO-YOLO VS YOLOv5

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**DAMO-YOLO**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv5**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 42.0 | N/A |
| s | 46.0 | 37.4 |
| m | 49.2 | 45.4 |
| l | 50.8 | 49.0 |
| x | N/A | 50.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**DAMO-YOLO**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv5**</span></center> |
|---------|-----------------------|-----------------------|
| n | 2.32 | N/A |
| s | 3.45 | 1.92 |
| m | 5.09 | 4.03 |
| l | 7.18 | 6.61 |
| x | N/A | 11.89 |
