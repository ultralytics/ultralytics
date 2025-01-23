---
---
# DAMO-YOLO VS RTDETRv2

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**DAMO-YOLO**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**RTDETRv2**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 42.0 | N/A |
| s | 46.0 | 48.1 |
| m | 49.2 | 51.9 |
| l | 50.8 | 53.4 |
| x | N/A | 54.3 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**DAMO-YOLO**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**RTDETRv2**</span></center> |
|---------|-----------------------|-----------------------|
| n | 2.32 | N/A |
| s | 3.45 | 5.03 |
| m | 5.09 | 7.51 |
| l | 7.18 | 9.76 |
| x | N/A | 15.03 |
