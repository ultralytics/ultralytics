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
DATA_PATH = f'{ROOT_DIR}/data/data'
TEST_DATA_PATH = f'{DATA_PATH}/test'
DATASET_DESCRIPTION = f'{DATA_PATH}/data.yaml'
PLOTS_PATH = f'{ROOT_DIR}/plots'
OS = sys.platform

DATASET_MEAN = (0.5227, 0.5336, 0.3736)
DATASET_STD = (0.2749, 0.2483, 0.1730)

ORIGINAL_CLASSES = [
    'anemone',
    'fisk',
    'bambuskorall',
    'vortesvamp',
    'sjostjerne',
    'risengrynkorall',
    'steinbit',
    'flyndre',
    'krabbe',
    'liten piperenser',
    'sjofaer',
    'skorpe',
    'Skate',
    'sjopolse',
    'krepsdyr',
    'Sjøtre',
    'stylocordyla',
    'sjobusk',
    'stor piperenser',
    'tare',
    'MRTSPB_svamp',
    'TVT_svamp',
    'finger_svamp',
    'blomkalkorall',
    'krakebolle',
    'snegle',
    'reirskjell',
    'Ukjent korall',
    'oyekorall',
    'soppel',
    'mysterie',
    'fjarsjolilje',
    'grandiflora'
    ]

# OPPDATERT FOR NYTT DATASETT
CLASSES_TO_KEEP = [
    'vortesvamp',
    'risengrynkorall',
    'liten piperenser',
    'sjofaer',
    'skorpe',
    'Sjøtre',
    'stylocordyla',
    'sjobusk',
    'tare',
    'MRTSPB_svamp',
    'TVT_svamp',
    'finger_svamp',
    'blomkalkorall',
    'oyekorall',
    'grandiflora',
    'anemone', 
    'fisk',
    'sjostjerne'
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