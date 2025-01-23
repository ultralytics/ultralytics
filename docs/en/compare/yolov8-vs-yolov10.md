---
---
# YOLOv8 VS YOLOv10

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv8**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv10**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 37.3 | 39.5 |
| s | 44.9 | 46.7 |
| m | 50.2 | 51.3 |
| b | N/A | 52.7 |
| l | 52.9 | 53.3 |
| x | 53.9 | 54.4 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv8**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv10**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.47 | 1.56 |
| s | 2.66 | 2.66 |
| m | 5.86 | 5.48 |
| b | N/A | 6.54 |
| l | 9.06 | 8.33 |
| x | 14.37 | 12.2 |
