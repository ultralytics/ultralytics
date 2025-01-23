---
---
# YOLOv10 VS YOLOv5

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv10**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv5**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 39.5 | N/A |
| s | 46.7 | 37.4 |
| m | 51.3 | 45.4 |
| b | 52.7 | N/A |
| l | 53.3 | 49.0 |
| x | 54.4 | 50.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv10**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv5**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.56 | N/A |
| s | 2.66 | 1.92 |
| m | 5.48 | 4.03 |
| b | 6.54 | N/A |
| l | 8.33 | 6.61 |
| x | 12.2 | 11.89 |
