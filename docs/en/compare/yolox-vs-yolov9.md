---
---
# YOLOX VS YOLOv9

## mAP Comparison

!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOX | mAP<sup>val<br>50<br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 37.8 |
		| s | 40.5 | 46.5 |
		| m | 46.9 | 51.5 |
		| l | 49.7 | 52.8 |
		| x | 51.1 | 55.1 |
		
## Speed Comparison

!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 2.3 |
		| s | 2.56 | 3.54 |
		| m | 5.43 | 6.43 |
		| l | 9.04 | 7.16 |
		| x | 16.1 | 16.77 |
		