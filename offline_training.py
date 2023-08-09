import os
import sys
from datetime import datetime

os.system("nohup sh -c '" + sys.executable + f" debug.py > ./training_baselines/res_{datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.txt 2>&1' &")