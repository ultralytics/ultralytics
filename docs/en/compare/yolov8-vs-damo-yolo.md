---
---
# YOLOv8 VS DAMO-YOLO

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv8**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**DAMO-YOLO**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 37.3 | 42.0 |
| s | 44.9 | 46.0 |
| m | 50.2 | 49.2 |
| l | 52.9 | 50.8 |
| x | 53.9 | N/A |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv8**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**DAMO-YOLO**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.47 | 2.32 |
| s | 2.66 | 3.45 |
| m | 5.86 | 5.09 |
| l | 9.06 | 7.18 |
| x | 14.37 | N/A |
