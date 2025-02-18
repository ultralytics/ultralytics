# RGB-IR-Ultralytics

## YOLO with Dual-modal Input

This project extends version 8.3.70 of [Ultralytics](https://github.com/ultralytics/ultralytics) by incorporating support for RGB+IR (visible light + infrared) dual-modal input for object detection tasks. Additionally, it provides an adjusted and adapted version of the [LLVIP](https://github.com/bupt-ai-cz/LLVIP) dataset, named [LLVIP-For-Ultralytics](https://github.com/Tdzdele/LLVIP-For-Ultralytics), for training and testing. Users are encouraged to cite the original work if utilized in research or applications.

**Note:** Dual-modal input training requires significantly more memory. If the training is automatically terminated with a "Killed" prompt, it may indicate a memory overflow. Monitor memory usage during training and adjust hyperparameters based on the device’s specifications.
