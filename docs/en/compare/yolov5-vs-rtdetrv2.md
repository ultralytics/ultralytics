---
---
# YOLOv5 VS RTDETRv2

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv5**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**RTDETRv2**</span></center> |
|----|----------------------------------|------------------------------------|
| s | 37.4 | 48.1 |
| m | 45.4 | 51.9 |
| l | 49.0 | 53.4 |
| x | 50.7 | 54.3 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv5**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**RTDETRv2**</span></center> |
|---------|-----------------------|-----------------------|
| s | 1.92 | 5.03 |
| m | 4.03 | 7.51 |
| l | 6.61 | 9.76 |
| x | 11.89 | 15.03 |
