---
---
# YOLOv5 VS YOLOv8

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv5**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv8**</span></center> |
|----|----------------------------------|------------------------------------|
| n | N/A | 37.3 |
| s | 37.4 | 44.9 |
| m | 45.4 | 50.2 |
| l | 49.0 | 52.9 |
| x | 50.7 | 53.9 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv5**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv8**</span></center> |
|---------|-----------------------|-----------------------|
| n | N/A | 1.47 |
| s | 1.92 | 2.66 |
| m | 4.03 | 5.86 |
| l | 6.61 | 9.06 |
| x | 11.89 | 14.37 |
