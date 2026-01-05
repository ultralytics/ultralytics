
# get the file path and juddge whether the folder name is ultralytics
current_dir=$(pwd)
base_dir=$(basename "$current_dir")
if [ "$base_dir" != "ultralytics" ]; then
    echo "Please run this script from the 'ultralytics' directory."
    exit 1
fi

# Source helper functions (set_conda_env, set_wandb_true, setup_experiment, run_training)
# Make sure these functions are defined before running this script
# source ./helper_functions.sh



tp_project_dir=runs/yoloe26_tp
vp_project_dir=runs/yoloe26_vp
pf_project_dir=runs/yoloe26_pf
seg_project_dir=runs/yoloe26_seg

tp_trainer="YOLOETrainerFromScratch"
vp_trainer="YOLOEVPTrainer"
pf_trainer="YOLOEPEFreeTrainer"
seg_trainer="YOLOESegmentTrainer"


train_tp_switch=true
train_vp_switch=false
train_pf_switch=false
train_seg_switch=false

tp_device="0,1,2,3"
vp_device="4,5,6"
pf_device="7"
seg_device="4,5"


#################################################### train tp ##################################################################
model=26x
weight_path="weights/yolo26x-objv1-seg[foryoloe].pt"
ptw="objv1"
clip_weight_name="mobileclip2:b" # mobileclip2b

ag=True

trainer=$tp_trainer

epo=15
close_mosaic=2
batch_size=256
device=$tp_device

optimizer="MuSGD"
lr0=0.00125
lrf=0.5
momentum=0.9
weight_decay=0.0005
o2m=1

copy_paste=0.20
mixup=0.60

project_dir=$tp_project_dir
tp_exp_name=${model}_ptw${ptw}_bs${batch_size}_epo${epo}_close${close_mosaic}_engine_or260105_tp
exp_name=$tp_exp_name

pyfile="finetune_yoloe26.py"
command="
python $pyfile \
    --model_version $model  --clip_weight_name $clip_weight_name --weight_path $weight_path \
    --ag $ag \
    --trainer $trainer \
    --optimizer $optimizer  --lr0 $lr0 --lrf $lrf  --momentum $momentum --weight_decay $weight_decay  --o2m $o2m \
    --epochs $epo  --close_mosaic $close_mosaic --batch $batch_size --device $device \
    --copy_paste $copy_paste --mixup $mixup  \
    --override or260105 \
    --project $project_dir --name $exp_name 
"

###################### lanch ##############################
screen_name=$exp_name

if [ "$train_tp_switch" = true ]; then
    set_conda_env ultra # Activate default environment
    set_wandb_true
    setup_experiment "$project_dir" "$exp_name" "$screen_name"  # checks and setups
    screen -S "$screen_name" bash -c "$(declare -f run_training); run_training '$statusfile' '$logfile' '$command'"
fi
#################################################### check tp best weight  ###############################################################
tp_best_weight=$tp_project_dir/$tp_exp_name/weights/best.pt

tp_best_weight="runs/yoloe26x_tp_ultra6/mobileclip2:b_26x_ptwyolo26x-objv1_bs256_epo30_close2_opMuSGD_o2m1_engine_tp/weights/best.pt"

if [ ! -f "$tp_best_weight" ]; then
    echo "TP best weight file not found at $tp_best_weight"
    exit 1
fi

#################################################### train segment head ###############################################################



model=26x
weight_path=$tp_best_weight
ptw="best"
clip_weight_name="mobileclip2:b" # mobileclip2b

ag=True

trainer=$seg_trainer

epo=15
close_mosaic=2
batch_size=256
device=$seg_device


optimizer="MuSGD"
lr0=0.00125
lrf=0.5
momentum=0.9
weight_decay=0.0005
o2m=1

copy_paste=0.15
mixup=0.05


project_dir=$seg_project_dir
exp_name=${model}_ptw${ptw}_bs${batch_size}_epo${epo}_close${close_mosaic}_engine_seg

pyfile="finetune_yoloe26.py"
command="
python $pyfile \
    --model_version $model  --clip_weight_name $clip_weight_name --weight_path $weight_path \
    --ag $ag \
    --trainer $trainer \
    --optimizer $optimizer  --lr0 $lr0 --lrf $lrf  --momentum $momentum --weight_decay $weight_decay  --o2m $o2m \
    --epochs $epo  --close_mosaic $close_mosaic --batch $batch_size --device $device \
    --copy_paste $copy_paste --mixup $mixup  \
    --project $project_dir --name $exp_name 
"
###################### lanch ############################## 
screen_name=$exp_name


if [ "$train_seg_switch" = true ]; then
    set_conda_env ultra # Activate default environment
    set_wandb_true
    setup_experiment "$project_dir" "$exp_name" "$screen_name"  # checks and setups
    screen -S "$screen_name" bash -c "$(declare -f run_training); run_training '$statusfile' '$logfile' '$command'"
fi
#################################################### train vp ###############################################################
model=26x
weight_path=$tp_best_weight
ptw="best"
clip_weight_name="mobileclip2:b" # mobileclip2b

ag=True

trainer=$vp_trainer

epo=10
close_mosaic=2
batch_size=256
device=$vp_device

optimizer="AdamW"
lr0=0.002
lrf=0.01
momentum=0.9
weight_decay=0.025
o2m=1

# optimizer="MuSGD"
# lr0=0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0005
# o2m=1

copy_paste=0.15
mixup=0.05


project_dir=$vp_project_dir
exp_name=${model}_ptw${ptw}_bs${batch_size}_epo${epo}_close${close_mosaic}_engine_vp

pyfile="finetune_yoloe26.py"
command="
python $pyfile \
    --model_version $model  --clip_weight_name $clip_weight_name --weight_path $weight_path \
    --ag $ag \
    --trainer $trainer \
    --optimizer $optimizer  --lr0 $lr0 --lrf $lrf  --momentum $momentum --weight_decay $weight_decay  --o2m $o2m \
    --epochs $epo  --close_mosaic $close_mosaic --batch $batch_size --device $device \
    --copy_paste $copy_paste --mixup $mixup  \
    --project $project_dir --name $exp_name 
"
###################### lanch ############################## 
screen_name=$exp_name
if [ "$train_vp_switch" = true ]; then
    set_conda_env ultra # Activate default environment
    set_wandb_true
    setup_experiment "$project_dir" "$exp_name" "$screen_name"  # checks and setups
    screen -S "$screen_name" bash -c "$(declare -f run_training); run_training '$statusfile' '$logfile' '$command'"
fi
#################################################### train pf ###############################################################

model=26x
weight_path=$tp_best_weight
ptw="best"
clip_weight_name="mobileclip2:b" # mobileclip2b

ag=True

trainer=$pf_trainer

epo=10
close_mosaic=2
batch_size=256
device=$pf_device

optimizer="AdamW"
lr0=0.002
lrf=0.01
momentum=0.9
weight_decay=0.025
o2m=1

# optimizer="MuSGD"
# lr0=0.00125
# lrf=0.5
# momentum=0.9
# weight_decay=0.0005
# o2m=1

copy_paste=0.15
mixup=0.05


project_dir=$pf_project_dir
exp_name=${model}_ptw${ptw}_bs${batch_size}_epo${epo}_close${close_mosaic}_engine_pf

pyfile="finetune_yoloe26.py"
command="
python $pyfile \
    --model_version $model  --clip_weight_name $clip_weight_name --weight_path $weight_path \
    --ag $ag \
    --trainer $trainer \
    --optimizer $optimizer  --lr0 $lr0 --lrf $lrf  --momentum $momentum --weight_decay $weight_decay  --o2m $o2m \
    --epochs $epo  --close_mosaic $close_mosaic --batch $batch_size --device $device \
    --copy_paste $copy_paste --mixup $mixup  \
    --project $project_dir --name $exp_name 
"
###################### lanch ############################## 
screen_name=$exp_name

if [ "$train_pf_switch" = true ]; then
    set_conda_env ultra # Activate default environment
    set_wandb_true
    setup_experiment "$project_dir" "$exp_name" "$screen_name"  # checks and setups
    screen -S "$screen_name" bash -c "$(declare -f run_training); run_training '$statusfile' '$logfile' '$command'"
fi
