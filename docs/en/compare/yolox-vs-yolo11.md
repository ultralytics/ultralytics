---
---
# YOLOX VS YOLO11

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOX**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLO11**</span></center> |
|----|----------------------------------|------------------------------------|
| n | N/A | 39.5 |
| s | 40.5 | 47.0 |
| m | 46.9 | 51.4 |
| l | 49.7 | 53.2 |
| x | 51.1 | 54.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOX**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLO11**</span></center> |
|---------|-----------------------|-----------------------|
| n | N/A | 1.55 |
| s | 2.56 | 2.63 |
| m | 5.43 | 5.27 |
| l | 9.04 | 6.84 |
| x | 16.1 | 12.49 |
