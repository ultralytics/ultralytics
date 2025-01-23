---
---
# YOLO11 VS YOLOv10

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLO11**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv10**</span></center> |
|----|----------------------------------|------------------------------------|
| n | 39.5 | 39.5 |
| s | 47.0 | 46.7 |
| m | 51.4 | 51.3 |
| b | N/A | 52.7 |
| l | 53.2 | 53.3 |
| x | 54.7 | 54.4 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLO11**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv10**</span></center> |
|---------|-----------------------|-----------------------|
| n | 1.55 | 1.56 |
| s | 2.63 | 2.66 |
| m | 5.27 | 5.48 |
| b | N/A | 6.54 |
| l | 6.84 | 8.33 |
| x | 12.49 | 12.2 |
