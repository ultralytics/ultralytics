---
---
# YOLO11 VS DAMO-YOLO

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLO11**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**DAMO-YOLO**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 39.5 | 42.0 |
| s | 47.0 | 46.0 |
| m | 51.4 | 49.2 |
| l | 53.2 | 50.8 |
| x | 54.7 | N/A |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLO11**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**DAMO-YOLO**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.55 | 2.32 |
| s | 2.63 | 3.45 |
| m | 5.27 | 5.09 |
| l | 6.84 | 7.18 |
| x | 12.49 | N/A |
