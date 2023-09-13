import os
import sys
from datetime import datetime

os.system("nohup sh -c '" + sys.executable + f" finetune.py > ./fine-tune-cdv1/logs/res_{datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.txt 2>&1' &")