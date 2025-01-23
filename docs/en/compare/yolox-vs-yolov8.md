---
---
# YOLOX VS Ultralytics YOLOv8

## mAP Comparison

!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOX | mAP<sup>val<br>50<br>YOLOv8 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 37.3 |
		| s | 40.5 | 44.9 |
		| m | 46.9 | 50.2 |
		| l | 49.7 | 52.9 |
		| x | 51.1 | 53.9 |
		
## Speed Comparison

!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 1.47 |
		| s | 2.56 | 2.66 |
		| m | 5.43 | 5.86 |
		| l | 9.04 | 9.06 |
		| x | 16.1 | 14.37 |
		