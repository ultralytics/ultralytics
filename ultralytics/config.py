import subprocess
import sys


def get_git_root():
    try:
        git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], universal_newlines=True).strip()
        return git_root
    except subprocess.CalledProcessError:
        return None

ROOT_DIR = get_git_root()
# DATASET_NAME = "My_dataset"
DATASET_NAME = "My_dataset_new"
DATA_PATH = f'{ROOT_DIR}/data_new'
LABELS_PATH = f'{DATA_PATH}/labels/labels.json'
TEST_DATA_PATH = f'{DATA_PATH}/test_split'
TEST_SET_DESCRIPTION = f'{ROOT_DIR}/test.yaml'
DATASET_DESCRIPTION = f'{DATA_PATH}/data.yaml'
PLOTS_PATH = f'{ROOT_DIR}/plots'
OS = sys.platform


DATASET_MEAN = (0.5227, 0.5336, 0.3736)
DATASET_STD = (0.2749, 0.2483, 0.1730)

ORIGINAL_CLASSES = [
    'anemone',
    'Massiv rund, tjukk skalformet, poros bulkeformet_svamp',
    'fisk',
    'blomkalkorall',
    'vortesvamp',
    'sjostjerne',
    'finger_svamp',
    'risengrynkorall',
    'Tynn vifteformet, traktformet_svamp',
    'steinbit',
    'flyndre',
    'krabbe',
    'liten piperenser',
    'sjofaer',
    'skorpe',
    'mysterie',
    'sjopolse',
    'krepsdyr',
    'stor piperenser',
    'tare',
    'krakebolle',
    'snegle',
    'soppel',
    'fjarsjolilje',
    'grandiflora',
    'bambuskorall',
    'Skate',
    'Sjøtre',
    'stylocordyla',
    'sjobusk',
    'reirskjell',
    'Ukjent korall',
    'oyekorall'
    ]

CLASSES_MAPPING = {
    "Massiv rund, tjukk skalformet, poros bulkeformet_svamp": "MRTSPB_svamp",
    "Tynn vifteformet, traktformet_svamp": "TVT_svamp",
}

CLASSES_TO_KEEP = [
    'anemone',
    'Massiv rund, tjukk skalformet, poros bulkeformet_svamp',
    'fisk',
    'blomkalkorall',
    'vortesvamp',
    'sjostjerne',
    'finger_svamp',
    'risengrynkorall',
    'Tynn vifteformet, traktformet_svamp',
    'flyndre',
    'liten piperenser',
    'sjofaer',
    'skorpe',
    'sjopolse',
    'krepsdyr',
    'stor piperenser',
    'tare',
    'krakebolle',
    'grandiflora',
    'bambuskorall',
    'Sjøtre',
    'stylocordyla',
    'sjobusk',
    'oyekorall'
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