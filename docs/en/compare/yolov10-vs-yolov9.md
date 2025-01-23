---
---
# YOLOv10 VS YOLOv9

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv10**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv9**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 39.5 | 37.8 |
| s | 46.7 | 46.5 |
| m | 51.3 | 51.5 |
| b | 52.7 | N/A |
| l | 53.3 | 52.8 |
| x | 54.4 | 55.1 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv10**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv9**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.56 | 2.3 |
| s | 2.66 | 3.54 |
| m | 5.48 | 6.43 |
| b | 6.54 | N/A |
| l | 8.33 | 7.16 |
| x | 12.2 | 16.77 |
