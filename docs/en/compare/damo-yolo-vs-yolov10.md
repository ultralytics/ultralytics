---
---
# DAMO-YOLO VS YOLOv10

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**DAMO-YOLO**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv10**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 42.0 | 39.5 |
| s | 46.0 | 46.7 |
| m | 49.2 | 51.3 |
| b | N/A | 52.7 |
| l | 50.8 | 53.3 |
| x | N/A | 54.4 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**DAMO-YOLO**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv10**</span></center> |
|---------|-----------------------|-----------------------|
| n | 2.32 | 1.56 |
| s | 3.45 | 2.66 |
| m | 5.09 | 5.48 |
| b | N/A | 6.54 |
| l | 7.18 | 8.33 |
| x | N/A | 12.2 |
