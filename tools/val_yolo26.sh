



yoloe26n_tp="runs/yoloe26_tp/26n_ptwobjv1_bs256_epo30_close2_engine_old_engine_data_tp[ultra8]/weights/best.pt"
yoloe26s_tp="runs/yoloe26_tp/26s_ptwobjv1_bs256_epo30_close2_engine_old_engine_data_tp[ultra8]/weights/best.pt"
yoloe26m_tp="runs/yoloe26_tp/26m_ptwobjv1_bs256_epo25_close2_engine_old_engine_data_tp[ultra8]/weights/best.pt"
yoloe26l_tp="runs/yoloe26_tp/26l_ptwobjv1_bs256_epo20_close2_engine_old_engine_data_tp[ultra6]/weights/best.pt"
yoloe26x_tp="runs/yoloe26_tp/26x_ptwobjv1_bs256_epo15_close2_engine_old_engine_data_tp[ultra6]/weights/best.pt"

yoloe26n_vp="runs/yoloe26_vp/26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt"
yoloe26s_vp="runs/yoloe26_vp/26s_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt"
yoloe26m_vp="runs/yoloe26_vp/26m_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt"
yoloe26l_vp="runs/yoloe26_vp/26l_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt"
yoloe26x_vp="runs/yoloe26_vp/26x_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt"

yoloe26n_seg="runs/yoloe26_seg/26n-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt"
yoloe26s_seg="runs/yoloe26_seg/26s-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt"
yoloe26m_seg="runs/yoloe26_seg/26m-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt"
yoloe26l_seg="runs/yoloe26_seg/26l-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt"
yoloe26x_seg="runs/yoloe26_seg/26x-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt"

yoloe26n_pf="runs/yoloe26_pf/26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt"
yoloe26s_pf="runs/yoloe26_pf/26s_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt"
yoloe26m_pf="runs/yoloe26_pf/26m_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt"
yoloe26l_pf="runs/yoloe26_pf/26l_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra6]/weights/best.pt"
yoloe26x_pf="runs/yoloe26_pf/26x_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra6]/weights/best.pt"


python ./tools/val_yoloe26_seg.py --device 0 \
 --model_weight "runs/yoloe26_seg/26n-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt" \
 --val_mode tp_only










python ./tools/val_yoloe26.py --device 0 \
 --model_weight  runs/yoloe26_vp/26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt \
 --val_mode all
# Validation Results (IoU=0.50:0.95): n
# ================================================================================
# tp_end2end          : mAP50-95=0.237, mAP50=0.329, P=0.166, R=0.150
# tp_not_end2end      : mAP50-95=0.247, mAP50=0.348, P=0.175, R=0.161
# vp_end2end          : mAP50-95=0.209, mAP50=0.296, P=0.133, R=0.179
# vp_not_end2end      : mAP50-95=0.219, mAP50=0.315, P=0.122, R=0.200



 python ./tools/val_yoloe26.py --device 0 \
 --model_weight  runs/yoloe26_vp/26s_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt \
 --val_mode all
# ================================================================================
# Validation Results (IoU=0.50:0.95):
# ================================================================================
# tp_end2end          : mAP50-95=0.299, mAP50=0.409, P=0.140, R=0.210
# tp_not_end2end      : mAP50-95=0.308, mAP50=0.423, P=0.157, R=0.195
# vp_end2end          : mAP50-95=0.271, mAP50=0.374, P=0.158, R=0.210
# vp_not_end2end      : mAP50-95=0.286, mAP50=0.399, P=0.210, R=0.234



python ./tools/val_yoloe26.py --device 0 \
 --model_weight  runs/yoloe26_vp/26m_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt \
 --val_mode all

# ================================================================================
# Validation Results (IoU=0.50:0.95):
# ================================================================================
# tp_end2end          : mAP50-95=0.354, mAP50=0.469, P=0.187, R=0.204
# tp_not_end2end      : mAP50-95=0.354, mAP50=0.476, P=0.156, R=0.243
# vp_end2end          : mAP50-95=0.313, mAP50=0.416, P=0.211, R=0.230
# vp_not_end2end      : mAP50-95=0.339, mAP50=0.456, P=0.244, R=0.262



python ./tools/val_yoloe26.py --device 0 \
 --model_weight  runs/yoloe26_vp/26l_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt \
 --val_mode all

# Validation Results (IoU=0.50:0.95):
# ================================================================================
# tp_end2end          : mAP50-95=0.368, mAP50=0.484, P=0.168, R=0.236
# tp_not_end2end      : mAP50-95=0.378, mAP50=0.500, P=0.166, R=0.243
# vp_end2end          : mAP50-95=0.337, mAP50=0.446, P=0.206, R=0.253
# vp_not_end2end      : mAP50-95=0.363, mAP50=0.485, P=0.267, R=0.278


python ./tools/val_yoloe26.py --device 0 \
 --model_weight  runs/yoloe26_vp/26x_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt \
 --val_mode all


# ================================================================================
# Validation Results (IoU=0.50:0.95):
# ================================================================================
# tp_end2end          : mAP50-95=0.395, mAP50=0.514, P=0.198, R=0.245
# tp_not_end2end      : mAP50-95=0.406, mAP50=0.532, P=0.180, R=0.256
# vp_end2end          : mAP50-95=0.362, mAP50=0.476, P=0.212, R=0.304
# vp_not_end2end      : mAP50-95=0.385, mAP50=0.507, P=0.276, R=0.305



python ./tools/val_yoloe26_pf.py --device 0 \
 --tp_weight  "runs/yoloe26_tp/26n_ptwobjv1_bs256_epo30_close2_engine_old_engine_data_tp[ultra8]/weights/best.pt" \
 --pf_weight  "runs/yoloe26_pf/26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt" \
 --single_cls False \
 --version 26n

#  18 classes had less than 10000 detections!
# Outputting 10000 detections for each class will improve AP further.
# If using detectron2, please use the lvdevil/infer_topk.py script to output a results file with 10000 detections for each class.
# ===
# [01/12 04:07:00] lvis.results WARNING: Assuming user provided the results in correct format.
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.000
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= -1 catIds=all] = 0.000
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= -1 catIds=all] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  r] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  c] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  f] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.000



python ./tools/val_yoloe26_pf.py --device 0 \
 --tp_weight  "runs/yoloe26_tp/26s_ptwobjv1_bs256_epo30_close2_engine_old_engine_data_tp[ultra8]/weights/best.pt" \
 --pf_weight  "runs/yoloe26_pf/26s_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt" \
 --single_cls False \
 --version 26s


# [01/12 04:15:31] lvis.results WARNING: Assuming user provided the results in correct format.
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.214
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= -1 catIds=all] = 0.286
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= -1 catIds=all] = 0.229
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.150
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.306
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.408
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  r] = 0.162
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  c] = 0.201
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  f] = 0.235
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.357
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.198
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.441
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.626

 python ./tools/val_yoloe26_pf.py --device 5 \
 --tp_weight  "runs/yoloe26_tp/26m_ptwobjv1_bs256_epo25_close2_engine_old_engine_data_tp[ultra8]/weights/best.pt" \
 --pf_weight  "runs/yoloe26_pf/26m_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt" \
 --single_cls False \
 --version 26m



# 1042 classes had less than 10000 detections!
# Outputting 10000 detections for each class will improve AP further.
# If using detectron2, please use the lvdevil/infer_topk.py script to output a results file with 10000 detections for each class.
# ===
# [01/12 04:18:52] lvis.results WARNING: Assuming user provided the results in correct format.
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.257
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= -1 catIds=all] = 0.336
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= -1 catIds=all] = 0.277
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.196
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.363
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.436
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  r] = 0.267
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  c] = 0.240
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  f] = 0.269
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.412
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.256
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.499
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.665

 


  python ./tools/val_yoloe26_pf.py --device 1 \
 --tp_weight  "runs/yoloe26_tp/26l_ptwobjv1_bs256_epo20_close2_engine_old_engine_data_tp[ultra6]/weights/best.pt" \
 --pf_weight  "runs/yoloe26_pf/26l_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra6]/weights/best.pt" \
 --single_cls False \
 --version 26l


# [01/12 02:48:28] lvis.results WARNING: Assuming user provided the results in correct format.
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.272
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= -1 catIds=all] = 0.354
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= -1 catIds=all] = 0.292
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.207
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.369
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.467
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  r] = 0.263
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  c] = 0.257
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  f] = 0.287
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.440
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.271
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.532
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.681
# copypaste: AP,AP50,AP75,APs,APm,APl,APr,APc,APf
# copypaste: 27.18,35.41,29.23,20.73,36.87,46.66,26.31,25.69,28.66



  python ./tools/val_yoloe26_pf.py --device 2 \
 --tp_weight  "runs/yoloe26_tp/26x_ptwobjv1_bs256_epo15_close2_engine_old_engine_data_tp[ultra6]/weights/best.pt" \
 --pf_weight  "runs/yoloe26_pf/26x_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra6]/weights/best.pt" \
 --single_cls False \
 --version 26x

# 1045 classes had less than 10000 detections!
# Outputting 10000 detections for each class will improve AP further.
# If using detectron2, please use the lvdevil/infer_topk.py script to output a results file with 10000 detections for each class.
# ===
# [01/12 04:09:19] lvis.results WARNING: Assuming user provided the results in correct format.
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.299
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= -1 catIds=all] = 0.387
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= -1 catIds=all] = 0.325
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.236
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.403
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.498
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  r] = 0.275
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  c] = 0.291
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  f] = 0.311
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.469
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.306
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.561
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.726
# copypaste: AP,AP50,AP75,APs,APm,APl,APr,APc,APf





 python ./tools/val_yoloe26_seg.py --device 6 \
    --model_weight  "runs/yoloe26_seg/26n-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt" \
    --version 26n \
    --batch 16


# Evaluating faster-coco-eval mAP using /home/shared/ultralytics/runs/segment/val26/predictions.json and ../datasets/lvis/annotations/lvis_v1_val.json...
# Evaluate annotation type *bbox*
# COCOeval_opt.evaluate() finished...
# DONE (t=92.89s).
# Accumulating evaluation results...
# COCOeval_opt.accumulate() finished...
# DONE (t=0.00s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.178
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.254
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.191
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.106
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.251
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.355
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  r] = 0.135
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  c] = 0.161
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  f] = 0.217
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 catIds=all] = 0.197
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 catIds=all] = 0.332
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.345
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.172
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.429
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.612
#  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.473
#  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.370
# Evaluate annotation type *segm*
# COCOeval_opt.evaluate() finished...
# DONE (t=177.18s).
# Accumulating evaluation results...
# COCOeval_opt.accumulate() finished...
# DONE (t=0.00s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.144
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.236
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.149
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.075
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.209
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.304
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  r] = 0.114
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  c] = 0.131
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  f] = 0.172
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 catIds=all] = 0.164
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 catIds=all] = 0.269
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.278
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.123
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.359
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.519
#  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.433
#  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.293


#  python ./tools/val_yoloe26_seg.py --device 5 \
#     --model_weight  "runs/yoloe26_seg/26s-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt" \
#     --version 26s \
#     --batch 16

# Evaluate annotation type *bbox*
# COCOeval_opt.evaluate() finished...
# DONE (t=69.80s).
# Accumulating evaluation results...
# COCOeval_opt.accumulate() finished...
# DONE (t=0.00s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.239
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.332


#  Evaluate annotation type *segm*
# COCOeval_opt.evaluate() finished...
# DONE (t=191.96s).
# Accumulating evaluation results...
# COCOeval_opt.accumulate() finished...
# DONE (t=0.00s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.197
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.309


 python ./tools/val_yoloe26_seg.py --device 7 \
    --model_weight  "runs/yoloe26_seg/26m-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt" \
    --version 26m \
    --batch 16

# Evaluating faster-coco-eval mAP using /home/shared/ultralytics/runs/segment/val24/predictions.json and ../datasets/lvis/annotations/lvis_v1_val.json...
# Evaluate annotation type *bbox*
# COCOeval_opt.evaluate() finished...
# DONE (t=87.68s).
# Accumulating evaluation results...
# COCOeval_opt.accumulate() finished...
# DONE (t=0.00s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.279
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.378
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.301
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.199
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.385
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.486
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  r] = 0.238
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  c] = 0.255
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  f] = 0.325
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 catIds=all] = 0.268
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 catIds=all] = 0.448
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.468
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.296
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.580
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.714
#  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.607
#  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.504
# Evaluate annotation type *segm*
# COCOeval_opt.evaluate() finished...
# DONE (t=159.36s).
# Accumulating evaluation results...
# COCOeval_opt.accumulate() finished...
# DONE (t=0.00s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.232
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.356
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.244
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.144
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.336
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.433
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  r] = 0.210
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  c] = 0.215
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  f] = 0.262
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 catIds=all] = 0.230
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 catIds=all] = 0.372
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.387
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.217
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.503
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.640
#  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.568
#  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.414


 python ./tools/val_yoloe26_seg.py --device 3 \
    --model_weight  "runs/yoloe26_seg/26l-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt" \
    --version 26l \
    --batch 32

# Evaluating faster-coco-eval mAP using /home/shared/ultralytics/runs/segment/val22/predictions.json and ../datasets/lvis/annotations/lvis_v1_val.json...
# Evaluate annotation type *bbox*
# COCOeval_opt.evaluate() finished...
# DONE (t=89.94s).
# Accumulating evaluation results...
# COCOeval_opt.accumulate() finished...
# DONE (t=0.00s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.295
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.395
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.318
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.214
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.395
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.498
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  r] = 0.253
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  c] = 0.271
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  f] = 0.340
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 catIds=all] = 0.275
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 catIds=all] = 0.466
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.487
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.312
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.606
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.740
#  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.625
#  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.526
# Evaluate annotation type *segm*
# ^[[B^[[BCOCOeval_opt.evaluate() finished...
# DONE (t=158.91s).
# Accumulating evaluation results...
# COCOeval_opt.accumulate() finished...
# DONE (t=0.00s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.243
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.372
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.259
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.153
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.340
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.435
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  r] = 0.214
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  c] = 0.229
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  f] = 0.272
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 catIds=all] = 0.234
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 catIds=all] = 0.385
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.401
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.226
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.520
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.651
#  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.586
#  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.430



 python ./tools/val_yoloe26_seg.py --device 3 \
    --model_weight  "runs/yoloe26_seg/26x-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt" \
    --version 26x \
    --batch 16



# Evaluating faster-coco-eval mAP using /home/shared/ultralytics/runs/segment/val20/predictions.json and ../datasets/lvis/annotations/lvis_v1_val.json...
# Evaluate annotation type *bbox*
# COCOeval_opt.evaluate() finished...
# DONE (t=88.70s).
# Accumulating evaluation results...
# COCOeval_opt.accumulate() finished...
# DONE (t=0.00s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.323
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.430
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.349
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.241
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.430
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.522
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  r] = 0.279
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  c] = 0.304
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  f] = 0.364
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 catIds=all] = 0.288
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 catIds=all] = 0.483
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.507
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.336
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.632
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.761
#  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.647
#  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.549
# Evaluate annotation type *segm*
# COCOeval_opt.evaluate() finished...
# DONE (t=173.41s).
# Accumulating evaluation results...
# COCOeval_opt.accumulate() finished...
# DONE (t=0.00s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.269
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.405
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.288
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.176
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.374
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.467
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  r] = 0.242
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  c] = 0.256
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=  f] = 0.294
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 catIds=all] = 0.248
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 catIds=all] = 0.404
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.422
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 catIds=all] = 0.246
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 catIds=all] = 0.550
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 catIds=all] = 0.685
#  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 catIds=all] = 0.608
#  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 catIds=all] = 0.457







yoloe26n_seg="runs/yoloe26_seg/26n-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt"
yoloe26s_seg="runs/yoloe26_seg/26s-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt"
yoloe26m_seg="runs/yoloe26_seg/26m-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt"
yoloe26l_seg="runs/yoloe26_seg/26l-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt"
yoloe26x_seg="runs/yoloe26_seg/26x-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt"




nohup python ./tools/val_yoloe26_seg.py --device 0 --model_weight "runs/yoloe26_seg/26n-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt" --val_mode tp_only > val_yoloe26_seg_26n_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# val_yoloe26_seg_26n_20260112_064149.log



nohup python ./tools/val_yoloe26_seg.py --device 0 --model_weight "runs/yoloe26_seg/26n-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt" --val_mode vp_only > val_yoloe26_seg_26n_vp_only_$(date +%Y%m%d_%H%M%S).log 2>&1 &



# s
nohup python ./tools/val_yoloe26_seg.py --device 1 --model_weight "runs/yoloe26_seg/26s-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt" --val_mode all > val_yoloe26_seg_26s_$(date +%Y%m%d_%H%M%S).log 2>&1 &



# m
nohup python ./tools/val_yoloe26_seg.py --device 2 --model_weight "runs/yoloe26_seg/26m-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt" --val_mode all > val_yoloe26_seg_26m_$(date +%Y%m%d_%H%M%S).log 2>&1 &



# l 
nohup python ./tools/val_yoloe26_seg.py --device 3 --model_weight "runs/yoloe26_seg/26l-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt" --val_mode all > val_yoloe26_seg_26l_$(date +%Y%m%d_%H%M%S).log 2>&1 &



# x
nohup python ./tools/val_yoloe26_seg.py --device 4 --model_weight "runs/yoloe26_seg/26x-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt" --val_mode all > val_yoloe26_seg_26x_$(date +%Y%m%d_%H%M%S).log 2>&1 &





