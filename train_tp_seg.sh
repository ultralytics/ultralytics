config_file="$HOME/.config/Ultralytics/settings.json"

# Check if the config file exists
if [ -f "$config_file" ]; then
    # Use a Python script to read, modify, and write back the JSON file
    python3 - <<END
import json
import os

config_file = "$config_file"

# Read the JSON file
with open(config_file, 'r') as f:
    data = json.load(f)

# Check and modify the 'wandb' value
if data.get('wandb') != True:
    data['wandb'] = True
    # Write the modified content back to the file
    with open(config_file, 'w') as f:
        # indent=4 keeps the file formatted for better readability
        json.dump(data, f, indent=4)
    print(f"Set 'wandb' to true in {config_file}")
else:
    print(f"'wandb' is already set to true in {config_file}")
END
else
    echo "Config file $config_file does not exist."
fi




# Initialize conda for bash shell
source ~/miniconda3/etc/profile.d/conda.sh
# Activate ultralytics environment
conda activate ultra

project_dir=runs/yoloe26s_tp_ultra6
mkdir -p $project_dir
semseg_loss=False

###############################################default args #######################################



# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="yolo26s-objv1.pt"
# trainer="YOLOESegTrainerFromScratch"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b
# ptw="object365v1" 


# lr0=0.002
# lrf=0.01
# optimizer="AdamW"
# momentum=0.9
# weight_decay=0.025
# o2m=0.1

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_lr0${lr0}_lrf${lrf}_bn_o2m${o2m}_vpseg
# device=4,5

#  /ultralytics/runs/yoloe26s_tp_seg_ultra6/mobileclip2:b_26s-seg_bs128_epo30_close2_lr00.002_lrf0.01_bn_o2m0.1_vpseg5


###############################################default args #######################################



# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="yolo26s-objv1.pt"
# trainer="YOLOESegTrainerFromScratch"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b
# ptw="object365v1" 


# optimizer="MuSGD"
# lr0=0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0007
# o2m=0.1

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_vpseg
# device=6,7



###############################################default args #######################################



# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="yolo26s-objv1.pt"
# trainer="YOLOESegTrainerFromScratch"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b
# ptw="object365v1" 


# optimizer="MuSGD"
# lr0=0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0007
# o2m=0.1

# semseg_loss=False

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segseg${semseg_loss}_vpseg
# device=0,1

# using the following command to check the log:\n tail -f -n 50 ./runs/20251212_080224.log
# Current screen: 2203801.train
# exp name: mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segsegFalse_vpseg7


###############################################default args #######################################
# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="yolo26s-objv1.pt"
# trainer="YOLOESegTrainerFromScratch"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b
# ptw="object365v1" 


# optimizer="MuSGD"
# lr0=0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0007
# o2m=0.1

# semseg_loss=False

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segseg${semseg_loss}_segment26_tp 
# device=4,5


# using the following command to check the log:\n tail -f -n 50 ./runs/20251215_031349.log
# Current screen: 2203801.train
# exp name: mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segsegFalse_segment26_tp

###############################################default args #######################################
# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="yolo26s-objv1.pt"
# trainer="YOLOESegTrainerFromScratch"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b
# ptw="object365v1" 


# optimizer="MuSGD"
# lr0=0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0007
# o2m=0.1

# semseg_loss=False

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segseg${semseg_loss}_segment26_mdata1_tp
# device=6,7
#mdata1  generate by Dec 17,2025
#  using the following command to check the log:
#  tail -f -n 50 ./runs/20251217_012050.log
#  Current screen: 393113.train3
#  exp name: mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segsegFalse_segment26_mdata1_tp
###############################################default args #######################################
# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="yolo26s-objv1.pt"
# trainer="YOLOESegTrainerFromScratch"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b
# ptw="object365v1" 


# optimizer="MuSGD"
# lr0=0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0007
# o2m=0.1

# semseg_loss=False

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segseg${semseg_loss}_segment26_mdata1_detachedproto_tp
# device=4,5

#  using the following command to check the log:
#  tail -f -n 50 ./runs/20251217_233701.log
#  Current screen: 2203801.train
#  exp name: mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segsegFalse_segment26_mdata1_detachedproto_tp

############################################### train only the seg head #######################################
# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="weights/yoloe-26s.pt"
# trainer="YOLOESegTrainerSegHead"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b
# ptw="object365v1" 


# optimizer="MuSGD"
# lr0=0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0007
# o2m=0.1

# semseg_loss=False

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segseg${semseg_loss}_segment26_mdata1_trainseghead_tp
# device=4,5
#  using the following command to check the log:
#  tail -f -n 50 ./runs/20251224_020343.log
#  Current screen: 117435.cuda45
#  exp name: mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segsegFalse_segment26_mdata1_trainseghead_tp2



############################################### train only the seg head #######################################
# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="weights/yoloe-26s.pt"
# trainer="YOLOESegTrainerFromScratch"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b
# ptw="object365v1" 


# optimizer="MuSGD"
# lr0=0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0007
# o2m=0.1

# semseg_loss=False

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segseg${semseg_loss}_segment_mdata1_tp
# device=6,7
#  using the following command to check the log:
#  tail -f -n 50 ./runs/20251224_030140.log
#  Current screen: 147470.cuda67
#  exp name: mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segsegFalse_segment_mdata1_tp4

############################################### train only the seg head #######################################
# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="weights/yolo26s-objv1-seg[foryoloe].pt"
# ptw="yolo26s-objv1-seg" 

# trainer="YOLOESegTrainerFromScratch"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b



# optimizer="MuSGD"
# lr0=0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0007
# o2m=0.1

# semseg_loss=False

# exp_name=${clip_weight_name}_${model}_ptw${ptw}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segseg${semseg_loss}_segment_mdata1_tp
# device=6,7

# Command: python ultralytics/finetune_yoloe26.py     --model_version 26s-seg     --lr0 0.00125     --lrf 0.5     --optimizer MuSGD     --momentum 0.9     --weight_decay 0.0007     --epochs 30     --close_mosaic 2     --batch 128     --device 6,7     --project runs/yoloe26s_tp_seg_ultra6     --name mobileclip2:b_26s-seg_ptwyolo26s-objv1-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segsegFalse_segment_mdata1_tp     --clip_weight_name mobileclip2:b     --ag True     --o2m 0.1     --weight_path weights/yolo26s-objv1-seg[foryoloe].pt     --trainer YOLOESegTrainerFromScratch     --semseg_loss False 
#  using the following command to check the log:
#  tail -f -n 50 ./runs/20251225_224317.log
#  Current screen: 147470.cuda67
#  exp name: mobileclip2:b_26s-seg_ptwyolo26s-objv1-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segsegFalse_segment_mdata1_tp

############################################### train only the seg head #######################################
# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="weights/yolo26s-objv1-seg[foryoloe].pt"
# ptw="yolo26s-objv1-seg" 

# trainer="YOLOESegTrainerFromScratch"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=255
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b



# optimizer="MuSGD"
# lr0=0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0007
# o2m=0.1

# semseg_loss=False

# exp_name=${clip_weight_name}_${model}_ptw${ptw}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segment_engine1_tp
# device=3,4,5

#--------------------------------- Training Started ---------------------------------
# Command: python ultralytics_pro/finetune_yoloe26.py     --model_version 26s-seg     --lr0 0.00125     --lrf 0.5     --optimizer MuSGD     --momentum 0.9     --weight_decay 0.0007     --epochs 30     --close_mosaic 2     --batch 255     --device 3,4,5     --project runs/yoloe26s_tp_seg_ultra6     --name mobileclip2:b_26s-seg_ptwyolo26s-objv1-seg_bs255_epo30_close2_opMuSGD_o2m0.1_segment_engine1_tp     --clip_weight_name mobileclip2:b     --ag True     --o2m 0.1     --weight_path weights/yolo26s-objv1-seg[foryoloe].pt     --trainer YOLOESegTrainerFromScratch     --semseg_loss False 
#  using the following command to check the log:
#  tail -f -n 50 ./runs/20251227_233445.log
#  Current screen: 
#  exp name: mobileclip2:b_26s-seg_ptwyolo26s-objv1-seg_bs255_epo30_close2_opMuSGD_o2m0.1_segment_engine1_tp2

############################################### train only the seg head #######################################
# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="weights/yoloe-26s.pt"
# ptw="yoloe26s"
# trainer="YOLOESegTrainerSegHead"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b


# optimizer="MuSGD"
# lr0=0.0025 # 0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0007
# o2m=0.1

# semseg_loss=False
# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segment26_engine1_trainseghead_tp
# device=0,1

# Command: python ultralytics_pro/finetune_yoloe26.py     --model_version 26s-seg     --lr0 0.0025     --lrf 0.5     --optimizer MuSGD     --momentum 0.9     --weight_decay 0.0007     --epochs 30     --close_mosaic 2     --batch 128     --device 0,1     --project runs/yoloe26s_tp_seg_ultra6     --name mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segment26_engine1_trainseghead_tp     --clip_weight_name mobileclip2:b     --ag True     --o2m 0.1     --weight_path weights/yoloe-26s.pt     --trainer YOLOESegTrainerSegHead     --semseg_loss False 
#  using the following command to check the log:
#  tail -f -n 50 ./runs/20251226_102717.log
#  Current screen: 337609.cuda01
#  exp name: mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segment26_engine1_trainseghead_tp
############################################### train only the seg head #######################################
# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="weights/yoloe-26s.pt"
# ptw="yoloe26s"
# trainer="YOLOESegTrainerSegHead"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b


# optimizer="MuSGD"
# lr0=0.0025 # 0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0005
# o2m=0.1

# semseg_loss=False
# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segment26_engine1_trainseghead_tp
# device=6,7
#--------------------------------- Training Started ---------------------------------
#Command: python ultralytics/finetune_yoloe26.py     --model_version 26s-seg     --lr0 0.0025     --lrf 0.5     --optimizer MuSGD     --momentum 0.9     --weight_decay 0.0005     --epochs 30     --close_mosaic 2     --batch 128     --device 6,7     --project runs/yoloe26s_tp_seg_ultra6     --name mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segment26_engine1_trainseghead_tp     --clip_weight_name mobileclip2:b     --ag True     --o2m 0.1     --weight_path weights/yoloe-26s.pt     --trainer YOLOESegTrainerSegHead     --semseg_loss False 
#  using the following command to check the log:
#  tail -f -n 50 ./runs/20251228_082403.log
#  Current screen: 147470.cuda67
#  exp name: mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segment26_engine1_trainseghead_tp

############################################### train only the seg head #######################################
# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="weights/yoloe-26s.pt"
# ptw="yoloe26s"
# trainer="YOLOESegTrainerSegHead"
# model=26s-seg
# epo=10
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b


# optimizer="MuSGD"
# lr0=0.0025 # 0.00125
# lrf=0.05
# momentum=0.9
# weight_decay=0.0005
# o2m=0.1

# semseg_loss=False
# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segment26_engine1_trainseghead_tp
# device=6,7

# --------------------------------- Training Started ---------------------------------
# Command: python ultralytics/finetune_yoloe26.py     --model_version 26s-seg     --lr0 0.0025     --lrf 0.05     --optimizer MuSGD     --momentum 0.9     --weight_decay 0.0005     --epochs 10     --close_mosaic 2     --batch 128     --device 6,7     --project runs/yoloe26s_tp_seg_ultra6     --name mobileclip2:b_26s-seg_bs128_epo10_close2_opMuSGD_o2m0.1_segment26_engine1_trainseghead_tp     --clip_weight_name mobileclip2:b     --ag True     --o2m 0.1     --weight_path weights/yoloe-26s.pt     --trainer YOLOESegTrainerSegHead     --semseg_loss False 
#  using the following command to check the log:
#  tail -f -n 50 ./runs/20251229_172715.log
#  Current screen: 147470.cuda67
#  exp name: mobileclip2:b_26s-seg_bs128_epo10_close2_opMuSGD_o2m0.1_segment26_engine1_trainseghead_tp

############################################### train only the seg head #######################################

# project_dir=runs/yoloe26s_tp_seg_ultra6
# weight_path="weights/yoloe-26s.pt"
# ptw="yoloe26s"
# trainer="YOLOESegTrainerSegHead"
# model=26s-seg
# epo=30
# close_mosaic=2
# batch_size=128
# ag=True

# clip_weight_name="mobileclip2:b" # mobileclip2b


# optimizer="MuSGD"
# lr0=0.0025 # 0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0005
# o2m=0.1

# semseg_loss=False
# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segment26_engine1_trainseghead_tp
# device=6,7


############################################### train only the seg head #######################################

project_dir=runs/yoloe26s_tp_seg_ultra6
weight_path="weights/yoloe-26s.pt"
ptw="yoloe26s"
trainer="YOLOESegTrainerSegHead"
model=26s-seg
epo=30
close_mosaic=2
batch_size=128
ag=True

clip_weight_name="mobileclip2:b" # mobileclip2b


optimizer="MuSGD"
lr0=0.0025 # 0.00125
lrf=0.5
momentum=0.9
weight_decay=0.0005
o2m=1

semseg_loss=False
exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_segment26_engine1_trainseghead_tp
device=6,7

# Command: python ultralytics/finetune_yoloe26.py     --model_version 26s-seg     --lr0 0.0025     --lrf 0.5     --optimizer MuSGD     --momentum 0.9     --weight_decay 0.0005     --epochs 30     --close_mosaic 2     --batch 128     --device 6,7     --project runs/yoloe26s_tp_seg_ultra6     --name mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m1_segment26_engine1_trainseghead_tp     --clip_weight_name mobileclip2:b     --ag True     --o2m 1     --weight_path weights/yoloe-26s.pt     --trainer YOLOESegTrainerSegHead     --semseg_loss False 
#  using the following command to check the log:
#  tail -f -n 50 ./runs/20251230_085700.log
#  Current screen: 
#  exp name: mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m1_segment26_engine1_trainseghead_tp
 ##############################################################################################
pyfile=ultralytics/finetune_yoloe26.py
timestamp=$(date +%Y%m%d_%H%M%S)
command="python $pyfile \
    --model_version $model \
    --lr0 $lr0 \
    --lrf $lrf \
    --optimizer $optimizer \
    --momentum $momentum \
    --weight_decay $weight_decay \
    --epochs $epo \
    --close_mosaic $close_mosaic \
    --batch $batch_size \
    --device $device \
    --project $project_dir \
    --name $exp_name \
    --clip_weight_name $clip_weight_name \
    --ag $ag \
    --o2m $o2m \
    --weight_path $weight_path \
    --trainer $trainer \
    --semseg_loss $semseg_loss "


nohup $command > "./runs/$timestamp.log" 2>&1 &

#  ##############################################################################################
echo "--------------------------------- Training Started ---------------------------------"
echo "Command: $command"
echo "#  using the following command to check the log:"
echo "#  tail -f -n 50 ./runs/$timestamp.log"
current_screen=$(echo $STY) # get the current screen 
echo "#  Current screen: $current_screen"
echo "#  exp name: $exp_name"



# todo. batch   close amp 
# --- IGNORE ---