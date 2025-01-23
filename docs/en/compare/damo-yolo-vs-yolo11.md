---
---
# DAMO-YOLO VS YOLO11

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**DAMO-YOLO**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLO11**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 42.0 | 39.5 |
| s | 46.0 | 47.0 |
| m | 49.2 | 51.4 |
| l | 50.8 | 53.2 |
| x | N/A | 54.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**DAMO-YOLO**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLO11**</span></center> |
|---------|-----------------------|-----------------------|
| n | 2.32 | 1.55 |
| s | 3.45 | 2.63 |
| m | 5.09 | 5.27 |
| l | 7.18 | 6.84 |
| x | N/A | 12.49 |
