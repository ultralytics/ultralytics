---
---
# YOLOv10 VS PP-YOLOE+

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv10**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**PP-YOLOE+**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 39.5 | 39.9 |
| s | 46.7 | 43.7 |
| m | 51.3 | 49.8 |
| b | 52.7 | N/A |
| l | 53.3 | 52.9 |
| x | 54.4 | 54.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv10**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**PP-YOLOE+**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.56 | 2.84 |
| s | 2.66 | 2.62 |
| m | 5.48 | 5.56 |
| b | 6.54 | N/A |
| l | 8.33 | 8.36 |
| x | 12.2 | 14.3 |
