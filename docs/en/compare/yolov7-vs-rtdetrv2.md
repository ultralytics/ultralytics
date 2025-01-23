---
---
# YOLOv7 VS RTDETRv2

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv7**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**RTDETRv2**</span></center> |
|----|----------------------------------|------------------------------------|
| s | N/A | 48.1 |
| m | N/A | 51.9 |
| l | 51.4 | 53.4 |
| x | 53.1 | 54.3 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv7**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**RTDETRv2**</span></center> |
|---------|-----------------------|-----------------------|
| s | N/A | 5.03 |
| m | N/A | 7.51 |
| l | 6.84 | 9.76 |
| x | 11.57 | 15.03 |
