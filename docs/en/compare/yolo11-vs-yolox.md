---
---
# YOLO11 VS YOLOX

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLO11**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOX**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 39.5 | N/A |
| s | 47.0 | 40.5 |
| m | 51.4 | 46.9 |
| l | 53.2 | 49.7 |
| x | 54.7 | 51.1 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLO11**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOX**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.55 | N/A |
| s | 2.63 | 2.56 |
| m | 5.27 | 5.43 |
| l | 6.84 | 9.04 |
| x | 12.49 | 16.1 |
