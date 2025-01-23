---
---

# PP-YOLOE+ VS YOLOv7

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**PP-YOLOE+**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**YOLOv7**</span></center> |
| ----------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| n           | 39.9                                                                                       | N/A                                                                                     |
| s           | 43.7                                                                                       | N/A                                                                                     |
| m           | 49.8                                                                                       | N/A                                                                                     |
| l           | 52.9                                                                                       | 51.4                                                                                    |
| x           | 54.7                                                                                       | 53.1                                                                                    |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**PP-YOLOE+**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**YOLOv7**</span></center> |
| ----------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| n           | 2.84                                                                                                               | N/A                                                                                                             |
| s           | 2.62                                                                                                               | N/A                                                                                                             |
| m           | 5.56                                                                                                               | N/A                                                                                                             |
| l           | 8.36                                                                                                               | 6.84                                                                                                            |
| x           | 14.3                                                                                                               | 11.57                                                                                                           |
