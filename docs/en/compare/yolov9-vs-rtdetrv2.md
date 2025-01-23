---
---
# YOLOv9 VS RTDETRv2

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv9**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**RTDETRv2**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 37.8 | N/A |
| s | 46.5 | 48.1 |
| m | 51.5 | 51.9 |
| l | 52.8 | 53.4 |
| x | 55.1 | 54.3 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv9**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**RTDETRv2**</span></center> |
|---------|-----------------------|-----------------------|
| n | 2.3 | N/A |
| s | 3.54 | 5.03 |
| m | 6.43 | 7.51 |
| l | 7.16 | 9.76 |
| x | 16.77 | 15.03 |
