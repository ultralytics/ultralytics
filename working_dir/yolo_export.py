from ultralytics.utils.benchmarks import ProfileModels
from ultralytics import RTDETR

# ProfileModels(paths=["yolo26n.pt"]).run()
# ProfileModels(paths=["yolo26s.pt"]).run()
# ProfileModels(paths=["yolo26m.pt"]).run()
# ProfileModels(paths=["yolo26l.pt"]).run()
# ProfileModels(paths=["yolo26x.pt"]).run()
# 
# ProfileModels(paths=["working_dir/yolo26nms_weights/yolo26n_nms.onnx"]).run()
# ProfileModels(paths=["working_dir/yolo26nms_weights/yolo26s_nms.onnx"]).run()
# ProfileModels(paths=["working_dir/yolo26nms_weights/yolo26m_nms.onnx"]).run()
# ProfileModels(paths=["working_dir/yolo26nms_weights/yolo26l_nms.onnx"]).run()
# ProfileModels(paths=["working_dir/yolo26nms_weights/yolo26x_nms.onnx"]).run()

# ProfileModels(paths=["yolo26n.onnx"]).run()
# ProfileModels(paths=["yolo26s.onnx"]).run()
# ProfileModels(paths=["yolo26m.onnx"]).run()
# ProfileModels(paths=["yolo26l.onnx"]).run()
# ProfileModels(paths=["yolo26x.onnx"]).run()

# ProfileModels(paths=["onnx_exports/rfdetr-nano/rfdetr-nano.onnx"], imgsz=384).run()
# ProfileModels(paths=["onnx_exports/rfdetr-small/rfdetr-small.onnx"], imgsz=512).run()
# ProfileModels(paths=["onnx_exports/rfdetr-medium/rfdetr-medium.onnx"], imgsz=576).run()
# ProfileModels(paths=["onnx_exports/rfdetr-large/rfdetr-large.onnx"], imgsz=704).run()
# ProfileModels(paths=["onnx_exports/rfdetr-xlarge/rfdetr-xlarge.onnx"], imgsz=700).run()
ProfileModels(paths=["onnx_exports/rfdetr-xxlarge/rfdetr-xxlarge.onnx"], imgsz=880).run()

# ProfileModels(paths=["output/lwdetr_tiny_coco/lwdetr_tiny.onnx"], imgsz=640).run()
# ProfileModels(paths=["output/lwdetr_small_coco/lwdetr_small.onnx"], imgsz=640).run()
# ProfileModels(paths=["output/lwdetr_medium_coco/lwdetr_medium.onnx"], imgsz=640).run()
# ProfileModels(paths=["output/lwdetr_large_coco/lwdetr_large.onnx"], imgsz=640).run()
# ProfileModels(paths=["output/lwdetr_xlarge_coco/lwdetr_xlarge.onnx"], imgsz=640).run()

# ProfileModels(paths=["deim_onnx_outputs/deim_dfine_hgnetv2_n_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_dfine_hgnetv2_s_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_dfine_hgnetv2_m_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_dfine_hgnetv2_l_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_dfine_hgnetv2_x_coco.onnx"], imgsz=640).run()

# ProfileModels(paths=["deim_onnx_outputs/deim_rtdetrv2_r18vd_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_rtdetrv2_r34vd_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_rtdetrv2_r50vd_m_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_rtdetrv2_r50vd_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_rtdetrv2_r101vd_coco.onnx"], imgsz=640).run()

# ProfileModels(paths=["deimv2/deimv2_hgnetv2_pico_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deimv2/deimv2_hgnetv2_n_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deimv2/deimv2_dinov3_s_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deimv2/deimv2_dinov3_m_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deimv2/deimv2_dinov3_l_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deimv2/deimv2_dinov3_x_coco.onnx"], imgsz=640).run()
