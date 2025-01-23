---
---
# DAMO-YOLO VS YOLOv8

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**DAMO-YOLO**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv8**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 42.0 | 37.3 |
| s | 46.0 | 44.9 |
| m | 49.2 | 50.2 |
| l | 50.8 | 52.9 |
| x | N/A | 53.9 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**DAMO-YOLO**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv8**</span></center> |
|---------|-----------------------|-----------------------|
| n | 2.32 | 1.47 |
| s | 3.45 | 2.66 |
| m | 5.09 | 5.86 |
| l | 7.18 | 9.06 |
| x | N/A | 14.37 |
