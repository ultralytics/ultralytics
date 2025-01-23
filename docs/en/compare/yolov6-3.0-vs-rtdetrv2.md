---
---
# YOLOv6-3.0 VS RTDETRv2

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv6-3.0**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**RTDETRv2**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 37.5 | N/A |
| s | 45.0 | 48.1 |
| m | 50.0 | 51.9 |
| l | 52.8 | 53.4 |
| x | N/A | 54.3 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv6-3.0**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**RTDETRv2**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.17 | N/A |
| s | 2.66 | 5.03 |
| m | 5.28 | 7.51 |
| l | 8.95 | 9.76 |
| x | N/A | 15.03 |
