---
---
# RTDETRv2 VS YOLO11

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**RTDETRv2**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLO11**</span></center> |
|----|----------------------------------|------------------------------------|
| n | N/A | 39.5 |
| s | 48.1 | 47.0 |
| m | 51.9 | 51.4 |
| l | 53.4 | 53.2 |
| x | 54.3 | 54.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**RTDETRv2**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLO11**</span></center> |
|---------|-----------------------|-----------------------|
| n | N/A | 1.55 |
| s | 5.03 | 2.63 |
| m | 7.51 | 5.27 |
| l | 9.76 | 6.84 |
| x | 15.03 | 12.49 |
