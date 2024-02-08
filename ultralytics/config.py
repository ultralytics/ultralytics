import subprocess
import sys


def get_git_root():
    try:
        git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], universal_newlines=True).strip()
        return git_root
    except subprocess.CalledProcessError:
        return None

ROOT_DIR = get_git_root()
DATASET_NAME = "My_dataset"
DATA_PATH = f'{ROOT_DIR}/data'
# DATA_PATH = "/media/vemund-sigurd/aa7e1253-187b-48d4-af32-eb9f050db5dd/vemund-sigurd/data/new_data"
# DATA_PATH = "/cluster/home/vemundtl/ultralytics/data"
DATA_PATH = "/Users/vemundlund/Code/Master/specialization_project/data"
LABELS_PATH = f'{DATA_PATH}/labels/labels.json'
TRAIN_DATA_PATH = f'{DATA_PATH}/train_split'
TEST_DATA_PATH = f'{DATA_PATH}/test_split'
VAL_DATA_PATH = f'{DATA_PATH}/val_split'
TEST_SET_DESCRIPTION = f'{ROOT_DIR}/test.yaml'
DATASET_DESCRIPTION = f'{DATA_PATH}/data.yaml'
PLOTS_PATH = f'{ROOT_DIR}/plots'
OS = sys.platform

CLASSES_TO_KEEP = [
    'grandiflora',
    'Tynn vifteformet, traktformet_svamp',
    'sjopolse',
    'sjofaer',
    'tare',
    'blomkalkorall',
    'sjostjerne',
    'Massiv rund, tjukk skalformet, poros bulkeformet_svamp',
    'skorpe',
    'liten piperenser',
    'anemone',
    'finger_svamp',
    'risengrynkorall',
    'fisk',
    'vortesvamp'
]

CLASSES_MAPPING = {
    "liten piperenser": "liten piperenser",
    "Massiv rund, tjukk skalformet, poros bulkeformet_svamp": "MRTSPB_svamp",
    "blomkalkorall": "blomkalkorall",
    "sjostjerne": "sjostjerne",
    "anemone": "anemone",
    "Tynn vifteformet, traktformet_svamp": "TVT_svamp",
    "vortesvamp": "vortesvamp",
    "fisk": "fisk",
    "finger_svamp": "finger_svamp",
    "risengrynkorall": "risengrynkorall",
    "tare": "tare",
    "sjopolse": "sjopolse",
    "skorpe": "skorpe",
    "sjofaer": "sjofaer",
    "grandiflora": "grandiflora",
}

DATASET_MEAN = (0.5227, 0.5336, 0.3736)
DATASET_STD = (0.2749, 0.2483, 0.1730)

MAX_BATCH = {
    "FasterRCNN_resnet50": 3,
    "FasterRCNN_resnet50_2": 2,
    "FasterRCNN_vitdet": 2,
    "FasterRCNN_squeezenet1_1": 8,
    "FasterRCNN_convnext_tiny": 2,
    "RetinaNet_resnet50": 4,
    "MaskRCNN_resnet50": 2,
    "MaskRCNN_convnext_tiny": 2,
    "yolo-nas": 8,
    }

#### EPOCHS ####
# Resnet50 -      10  min per epoch
# Squeezenet1_1 - 3   min per epoch 
# Vitdet -        12  min per epoch
# Retinanet -     8   min per epoch
# Convnext Faster -      25  min per epoch
# YOLOv8 -        3-4 min per epoch
# RTDETR -        30  min per epoch
# MASK RCNN -     13  min per epoch

ORIGINAL_CLASSES = [
    "anemone",
    "Massiv rund, tjukk skalformet, poros bulkeformet_svamp",
    "fisk",
    "blomkalkorall",
    "vortesvamp",
    "sjostjerne",
    "finger_svamp",
    "risengrynkorall",
    "Tynn vifteformet, traktformet_svamp",
    "steinbit",
    "flyndre",
    "krabbe",
    "liten piperenser",
    "sjofaer",
    "skorpe",
    "mysterie",
    "sjopolse",
    "krepsdyr",
    "stor piperenser",
    "tare",
    "krakebolle",
    "snegle",
    "soppel",
    "fjarsjolilje",
    "grandiflora"
]

def get_yolo_classes():
    indexes = []
    for i in range(len(ORIGINAL_CLASSES)):
        if ORIGINAL_CLASSES[i] in CLASSES_TO_KEEP:
            indexes.append(i)
    return indexes

def SPONGEBOB(model_name, word="TRAINING MODEL"):
    return rf"""
            {word} 
        .--..--..--..--..--..--.
        .' \  (`._   (_)     _   \
    .'    |  '._)         (_)  |
    \ _.')\      .----..---.   /
    |(_.'  |    /    .-\-.  \  |
    \     0|    |   ( O| O) | o|
    |  _  |  .--.____.'._.-.  |
    \ (_) | o         -` .-`  |
        |    \   |`-._ _ _ _ _\ /
        \    |   |  `. |_||_|   |    {model_name.upper()}
        | o  |    \_      \     |     -.   .-.
        |.-.  \     `--..-'   O |     `.`-' .'
    _.'  .' |     `-.-'      /-.__   ' .-'
    .' `-.` '.|='=.='=.='=.='=|._/_ `-'.'
    `-._  `.  |________/\_____|    `-.'
    .'   ).| '=' '='\/ '=' |
    `._.`  '---------------'
            //___\   //___\
                ||       ||
                ||_.-.   ||_.-.
                (_.--__) (_.--__)
    """