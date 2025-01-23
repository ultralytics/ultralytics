---
---
# YOLOv10 VS YOLOv6-3.0

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv10**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv6-3.0**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 39.5 | 37.5 |
| s | 46.7 | 45.0 |
| m | 51.3 | 50.0 |
| b | 52.7 | N/A |
| l | 53.3 | 52.8 |
| x | 54.4 | N/A |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv10**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv6-3.0**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.56 | 1.17 |
| s | 2.66 | 2.66 |
| m | 5.48 | 5.28 |
| b | 6.54 | N/A |
| l | 8.33 | 8.95 |
| x | 12.2 | N/A |
