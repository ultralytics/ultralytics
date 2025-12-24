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

############################################################################################. | 
# project_dir=runs/yoloe26s_vp_ultra6
# weight_path="./runs/yoloe26s_tp_ultra6/mobileclip2:b_26s_bs128_ptwobject365v1_close2_agdata2_lrf0.5_bn_exp/weights/best.pt"
# trainer="YOLOEVPTrainer"
# model=26s
# epo=10
# close_mosaic=2
# batch_size=128
# ag=True
# clip_weight_name="mobileclip2:b" # mobileclip2b
# ptw="object365v1" 
# optimizer="AdamW"
# lr0=0.002
# lrf=0.01
# momentum=0.9
# weight_decay=0.025
# o2m=0.1
# exp_name=${clip_weight_name}_${model}_bs${batch_size}_ptw${ptw}_close${close_mosaic}_agdata2_lrf${lrf}_bn_o2m${o2m}_newsavpe_vp
# device=0,1

# using the following command to check the log:
# tail -f -n 50 ./runs/20251218_012053.log
# Current screen: 359138.train2
# exp name: mobileclip2:b_26s_bs128_ptwobject365v1_close2_agdata2_lrf0.01_bn_o2m0.1_newsavpe_vp

############################################################################################. | 
project_dir=runs/yoloe26s_vp_ultra6
weight_path="./runs/yoloe26s_tp_ultra6/mobileclip2:b_26s_bs128_ptwobject365v1_close2_agdata2_lrf0.5_bn_exp/weights/best.pt"
trainer="YOLOEVPTrainer"
model=26s
epo=10
close_mosaic=2
batch_size=128
ag=True
clip_weight_name="mobileclip2:b" # mobileclip2b
ptw="object365v1" 

optimizer="MuSGD"
lr0=0.00125
lrf=0.5
momentum=0.9
weight_decay=0.0007
o2m=0.1

exp_name=${clip_weight_name}_${model}_bs${batch_size}_ptw${ptw}_close${close_mosaic}_agdata2_optim${optimizer}_bn_o2m${o2m}_vp
device=2,3

# using the following command to check the log:
#   tail -f -n 50 ./runs/20251224_091127.log
#   Current screen: 326335.cuda23
#   exp name: mobileclip2:b_26s_bs128_ptwobject365v1_close2_agdata2_optimMuSGD_bn_o2m0.1_vp
###############################################default args #######################################


pyfile=ultralytics/finetune_yoloe26.py

timestamp=$(date +%Y%m%d_%H%M%S)
nohup python $pyfile \
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
    > "./runs/$timestamp.log" 2>&1 &

##############################################################################################
echo "# using the following command to check the log:"
echo "#   tail -f -n 50 ./runs/$timestamp.log"
current_screen=$(echo $STY) # get the current screen 
echo "#   Current screen: $current_screen"
echo "#   exp name: $exp_name"







# todo. batch   close amp 
# --- IGNORE ---