---
---
# YOLOv5 VS PP-YOLOE+

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv5**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**PP-YOLOE+**</span></center> |
|----|----------------------------------|------------------------------------|
| n | N/A | 39.9 |
| s | 37.4 | 43.7 |
| m | 45.4 | 49.8 |
| l | 49.0 | 52.9 |
| x | 50.7 | 54.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv5**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**PP-YOLOE+**</span></center> |
|---------|-----------------------|-----------------------|
| n | N/A | 2.84 |
| s | 1.92 | 2.62 |
| m | 4.03 | 5.56 |
| l | 6.61 | 8.36 |
| x | 11.89 | 14.3 |
