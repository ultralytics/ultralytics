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



###############################################default args #######################################



# project_dir=runs/yoloe26s_tp_ultra6
# weight_path="yolo26s-objv1.pt"
# trainer="YOLOETrainerFromScratch"
# model=26s
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

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_tp
# device=4,5


# using the following command to check the log:\n tail -f -n 50 ./runs/20251212_065257.log
# Current screen: 393113.train3
# exp name: mobileclip2:b_26s_bs128_epo30_close2_opMuSGD_o2m0.1_tp3

###############################################default args #######################################



# project_dir=runs/yoloe26s_tp_ultra6
# weight_path="yolo26s-objv1.pt"
# trainer="YOLOETrainerFromScratch"
# model=26s
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

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_nmdata_tp
# device=6,7

# using the following command to check the log:\n tail -f -n 50 ./runs/20251213_103502.log
# Current screen: 1206388.train4
# exp name: mobileclip2:b_26s_bs128_epo30_close2_opMuSGD_o2m0.1_nmdata_tp

###############################################default args #######################################



# project_dir=runs/yoloe26s_tp_ultra6
# weight_path="yolo26s-objv1.pt"
# trainer="YOLOETrainerFromScratch"
# model=26s
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

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_nmdata-beta_tp
# device=6,7

# nmdata: using dataset cache without mask
# nmdata-beta: add flickr data 
# CACHE_SUFFIX=".engine.cache"  in. datasets.py
# using the following command to check the log:\n tail -f -n 50 ./runs/20251214_214854.log
# Current screen: 1206388.train4
# exp name: mobileclip2:b_26s_bs128_epo30_close2_opMuSGD_o2m0.1_nmdata-beta_tp

###############################################default args #######################################



# project_dir=runs/yoloe26s_tp_ultra6
# weight_path="yolo26s-objv1.pt"
# trainer="YOLOETrainerFromScratch"
# model=26s
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

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_maskdata_tp
# device=2,3
# using the following command to check the log:\n tail -f -n 50 ./runs/20251216_062128.log
# Current screen: 359138.train2
# exp name: mobileclip2:b_26s_bs128_epo30_close2_opMuSGD_o2m0.1_maskdata_tp
###############################################default args #######################################



# project_dir=runs/yoloe26m_tp_ultra6
# weight_path="yolo26m-objv1.pt"
# trainer="YOLOETrainerFromScratch"
# model=26m
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
# weight_decay=0.0005
# o2m=0.1

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_maskdata_tp
# device=2,3

# using the following command to check the log:
# tail -f -n 50 ./runs/20251221_105109.log
# Current screen: 2203801.train
# exp name: mobileclip2:b_26m_bs128_epo30_close2_opMuSGD_o2m0.1_maskdata_tp
###############################################default args #######################################



project_dir=runs/yoloe26s_tp_ultra6
weight_path="yolo26s-objv1.pt"
ptw="yolo26s-objv1"
trainer="YOLOETrainerFromScratch"
model=26s
epo=30
close_mosaic=2
batch_size=255
ag=True

clip_weight_name="mobileclip2:b" # mobileclip2b


optimizer="MuSGD"
lr0=0.00125
lrf=0.5
momentum=0.9
weight_decay=0.0007
o2m=0.1

copy_paste=0.15
mixup=0.05

exp_name=${clip_weight_name}_${model}_ptw${ptw}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_engine1_tp
device=0,1,2


# python commnand:
# python ultralytics_pro/finetune_yoloe26.py     --model_version 26s     --lr0 0.00125     --lrf 0.5     --optimizer MuSGD     --momentum 0.9     --weight_decay 0.0007     --epochs 30     --close_mosaic 2     --batch 255     --device 0,1,2     --project runs/yoloe26s_tp_ultra6     --name mobileclip2:b_26s_ptwyolo26s-objv1_bs255_epo30_close2_opMuSGD_o2m0.1_engine1_tp     --clip_weight_name mobileclip2:b     --ag True     --o2m 0.1     --weight_path yolo26s-objv1.pt     --trainer YOLOETrainerFromScratch     --copy_paste 0.15     --mixup 0.05 
# using the following command to check the log:
# tail -f -n 50 ./runs/20251227_091212.log
# Current screen: 1154717.cuda01-2
###############################################default args #######################################


# project_dir=runs/yoloe26l_tp_ultra6
# weight_path="yolo26l-objv1.pt"
# trainer="YOLOETrainerFromScratch"
# model=26l
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
# weight_decay=0.0005
# o2m=0.1

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_maskdata_tp
# device=4,5

# using the following command to check the log:
# tail -f -n 50 ./runs/20251221_105843.log
# Current screen: 359138.train2
# exp name: mobileclip2:b_26l_bs128_epo30_close2_opMuSGD_o2m0.1_maskdata_tp

###############################################default args #######################################


# project_dir=runs/yoloe26l_tp_ultra6
# weight_path="yolo26l-objv1.pt"
# trainer="YOLOETrainerFromScratch"
# model=26l
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
# weight_decay=0.0005
# o2m=0.1

# copy_paste=0.5
# mixup=0.15

# exp_name=${clip_weight_name}_${model}_bs${batch_size}_epo${epo}_close${close_mosaic}_op${optimizer}_o2m${o2m}_cp50_mix15_tp
# device=6,7


# using the following command to check the log:
# tail -f -n 50 ./runs/20251221_110801.log
# Current screen: 393113.train3
# exp name: mobileclip2:b_26l_bs128_epo30_close2_opMuSGD_o2m0.1_cp50_mix15_tp

###############################################default args #######################################


pyfile=ultralytics_pro/finetune_yoloe26.py
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
    --copy_paste $copy_paste \
    --mixup $mixup "

nohup $command > "./runs/$timestamp.log" 2>&1 &

##############################################################################################
echo "# python commnand:"
echo "# $command"

echo "# using the following command to check the log:"
echo "# tail -f -n 50 ./runs/$timestamp.log"
current_screen=$(echo $STY) # get the current screen 
echo "# Current screen: $current_screen"







# todo. batch   close amp 
# --- IGNORE ---