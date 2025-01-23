---
---
# RTDETRv2 VS YOLOv9

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**RTDETRv2**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv9**</span></center> |
|----|----------------------------------|------------------------------------|
| n | N/A | 37.8 |
| s | 48.1 | 46.5 |
| m | 51.9 | 51.5 |
| l | 53.4 | 52.8 |
| x | 54.3 | 55.1 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**RTDETRv2**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv9**</span></center> |
|---------|-----------------------|-----------------------|
| n | N/A | 2.3 |
| s | 5.03 | 3.54 |
| m | 7.51 | 6.43 |
| l | 9.76 | 7.16 |
| x | 15.03 | 16.77 |
