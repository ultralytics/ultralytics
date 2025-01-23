---
---
# RTDETRv2 VS DAMO-YOLO

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**RTDETRv2**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**DAMO-YOLO**</span></center> |
|----|----------------------------------|------------------------------------|
| n | N/A | 42.0 |
| s | 48.1 | 46.0 |
| m | 51.9 | 49.2 |
| l | 53.4 | 50.8 |
| x | 54.3 | N/A |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**RTDETRv2**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**DAMO-YOLO**</span></center> |
|---------|-----------------------|-----------------------|
| n | N/A | 2.32 |
| s | 5.03 | 3.45 |
| m | 7.51 | 5.09 |
| l | 9.76 | 7.18 |
| x | 15.03 | N/A |
