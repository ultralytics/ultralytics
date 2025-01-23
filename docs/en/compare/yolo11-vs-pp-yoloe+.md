---
---
# YOLO11 VS PP-YOLOE+

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLO11**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**PP-YOLOE+**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 39.5 | 39.9 |
| s | 47.0 | 43.7 |
| m | 51.4 | 49.8 |
| l | 53.2 | 52.9 |
| x | 54.7 | 54.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLO11**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**PP-YOLOE+**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.55 | 2.84 |
| s | 2.63 | 2.62 |
| m | 5.27 | 5.56 |
| l | 6.84 | 8.36 |
| x | 12.49 | 14.3 |
