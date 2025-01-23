---
---
# YOLOv5 VS YOLOv9

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv5**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv9**</span></center> |
|----|----------------------------------|------------------------------------|
| n | N/A | 37.8 |
| s | 37.4 | 46.5 |
| m | 45.4 | 51.5 |
| l | 49.0 | 52.8 |
| x | 50.7 | 55.1 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv5**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv9**</span></center> |
|---------|-----------------------|-----------------------|
| n | N/A | 2.3 |
| s | 1.92 | 3.54 |
| m | 4.03 | 6.43 |
| l | 6.61 | 7.16 |
| x | 11.89 | 16.77 |
