---
---
# RTDETRv2 VS PP-YOLOE+

## mAP Comparison

| **Variant** | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**RTDETRv2**</span></center> | <center><span style='width: 400px;'>**mAP<sup>val<br>50**<br>**PP-YOLOE+**</span></center> |
|----|----------------------------------|------------------------------------|
| n | N/A | 39.9 |
| s | 48.1 | 43.7 |
| m | 51.9 | 49.8 |
| l | 53.4 | 52.9 |
| x | 54.3 | 54.7 |

## Speed Comparison

| **Variant** | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**RTDETRv2**</span></center> | <center><span style='width: 200px;'>**Speed**<br><sup>T4 TensorRT10<br>(ms)</sup><br>**PP-YOLOE+**</span></center> |
|---------|-----------------------|-----------------------|
| n | N/A | 2.84 |
| s | 5.03 | 2.62 |
| m | 7.51 | 5.56 |
| l | 9.76 | 8.36 |
| x | 15.03 | 14.3 |
