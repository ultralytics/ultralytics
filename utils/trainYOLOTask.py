'''
@FileName   :trainYOLOTask.py
@Description:
@Date       :2025/02/21 16:38:32
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import re
import psutil
import datetime
import subprocess
from redis import Redis

from config import data_cfg, train_result, pretrained_models, train_result_topic_name
from utils import find_pt


class trainYOLODetectTask:
    '''
    yolo目标检测训练类
    '''

    def __init__(self, rds: Redis, taskId: str, taskName: str, taskNameList: list, taskProcDict: dict, labels: list, netType: str, trainType: str, modelId: str, paramters: dict, call_back):
        super().__init__()
        self.daemon = True
        self.labels = labels
        self.netType = netType
        self.trainType = trainType
        self.modelId = modelId
        self.rds = rds
        self.taskId = taskId
        self.taskName = taskName
        self.taskNameList = taskNameList
        self.taskProcDict = taskProcDict
        self.paramters = paramters
        self.func_call_back = call_back  # 通用的回调函数属性

    def generate_dir(self):
        if not os.path.exists(data_cfg):
            os.makedirs(data_cfg)
        elif not os.path.exists(train_result):
            os.makedirs(train_result)
        elif not os.path.exists(pretrained_models):
            os.makedirs(pretrained_models)

    def regnix_format(self):
        re_list = []
        r1 = r"(\d+)/"
        re_list.append(f"{r1}{self.paramters['epochs']}")
        r1 = r"^      Epoch"
        re_list.append(f"{r1}")
        r1 = r"^                 Class"
        re_list.append(f"{r1}")
        r1 = r"^                   all"
        re_list.append(f"{r1}")
        for i in range(len(self.labels)):
            re_list.append(f'^                   {self.labels[i]}')
        re_format = "|".join(re_list)
        print(f"re_format:{re_format}")
        return re_format
    # v2版本直接用ultralytics库,就不用再生成train.py脚本了

    def generate_train_script(self):
        train_params = {
            'data': f'{data_cfg}/{self.taskId}_{self.taskName}.yaml',
            'project': f'{train_result}',
            'name': f'{self.taskId}_{self.taskName}',
            'task': 'detect'
        }
        file = open('./train.py', 'w', encoding='utf-8')
        file.write("# 此文件由程序自动生成.\n")
        file.write("# 请勿手动修改.\n")

        # 开始写入训练代码
        file.write('import os\n')
        file.write('import shutil\n')
        # file.write('import datetime\n')
        file.write('from ultralytics import YOLO\n')
        if self.trainType == 'Init':  # 初始化训练
            if self.netType.startswith('yolov5'):
                file.write(
                    f"model_cfg = '{pretrained_models}/{self.netType}u.yaml'\n")
                file.write(
                    f"model_path = '{pretrained_models}/{self.netType}u.pt'\n")
            else:
                file.write(
                    f"model_cfg = '{pretrained_models}/{self.netType}.yaml'\n")
                file.write(
                    f"model_path = '{pretrained_models}/{self.netType}.pt'\n")
        else:  # 迭代--modelId是上一次训练的taskId_taskName
            iter_pre_model = find_pt('export_results', self.modelId)
            if self.netType.startswith('yolov5'):
                file.write(
                    f"model_cfg = '{pretrained_models}/{self.netType}u.yaml'\n")
                file.write(
                    f"model_path = '{iter_pre_model}'\n")
            else:
                file.write(
                    f"model_cfg = '{pretrained_models}/{self.netType}.yaml'\n")
                file.write(
                    f"model_path = '{iter_pre_model}'\n")

        for key, value in self.paramters.items():
            # 使用repr()确保value被合法转换
            # file.write(f"{key} = {repr(value)}\n")
            # file.write(f"{key} = {value}\n")
            train_params[key] = value
        file.write(f'train_params={train_params}\n')
        file.write('model = YOLO(model_cfg).load(model_path)\n')
        file.write('model.train(**train_params)\n')
        # 训练完成后的，模型移动操作代码块
        model_trained_path = '/'.join([train_params['project'],
                                      train_params['name'], 'weights', 'best.pt'])
        file.write(f"model_trained_path = '{model_trained_path}'\n")
        file.write(f"if not os.path.exists('{'export_results'}'):\n")
        file.write(f"    os.makedirs('{'export_results'}')\n")
        # model_rename = f"datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_{self.taskId}_{self.taskName}'"
        file.write(f"model_rename = '{self.taskId}_{self.taskName}'\n")
        file.write("model_rename = ''.join([model_rename,'.pt'])\n")
        file.write(
            f"model_rename_path = '/'.join(['export_results',model_rename])\n")
        file.write("shutil.copy(model_trained_path,model_rename_path)\n")
        file.close()

    def train(self, re_fmt):
        self.rds.hset(f'{self.taskId}_{self.taskName}_train_progress', mapping={
                      'status': 1, 'epoch': 0})
        log_file = open(
            f'./log/train_log_{self.taskId}_{self.taskName}.txt', 'w', encoding='utf-8')
        # Logger()--start_train_taskId_taskName.txt--记录进程、训练起始等中间信息(用于debug)*******
        # train_command = ["source activate paddle && python train.py"]
        train_command = ["python train.py"]  # 镜像中没有conda虚拟环境,只有base环境,但是包是全的就行
        process = subprocess.Popen(args=train_command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   encoding='utf-8',
                                   errors='replace',
                                   shell=True,
                                   executable='/bin/bash')
        # 回调函数
        create_time = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S.%f')[:-3]
        if not psutil.pid_exists(process.pid):  # 后台程序出错
            self.func_call_back('start', self.taskId, self.taskName, 0,
                                f'当前训练任务:{self.taskName}启动失败,后台进程已终止.', create_time)
            taskNameIndex = self.taskNameList.index(self.taskName)
            del self.taskNameList[taskNameIndex]  # 删除对应的任务
            self.rds.hdel(f'{self.taskId}_{self.taskName}_train_progress', *
                          self.rds.hkeys(f'{self.taskId}_{self.taskName}_train_progress'))
            log_file.close()
        else:  # 正常启动
            self.func_call_back('start', self.taskId, self.taskName, 1,
                                f'当前训练任务:{self.taskId}_{self.taskName}启动成功,后台进程已启动.', create_time)
            self.taskProcDict[self.taskName] = process  # self.train_task_obj
            for line in process.stdout:
                log_file.write(line)  # 这里看需求写入文档,目前只传epoch p r mAP数据传送到Redis
                log_file.flush()
                match_ret = re.search(re_fmt, line)
                # print("match:\t", match_ret)
                if match_ret:
                    if not match_ret.group(1):
                        print(line)
                    else:
                        print(line)
                        current_epoch = int(match_ret.group(1))
                        # 通过Redis发送中间训练过程
                        self.rds.hset(f'{self.taskId}_{self.taskName}_train_progress',
                                      mapping={'status': 1, 'epoch': current_epoch})
                        # 可能还会有其他训练过程信息发送....
            log_file.close()
            returncode = process.wait()  # 会一直到等到当前训练结束才返回0--正常结束;1--异常返回
            status = 0 if returncode == 0 else -1
            self.rds.hset(f'{self.taskId}_{self.taskName}_train_progress',
                          mapping={'status': status, 'epoch': 0})
            if not psutil.pid_exists(process.pid):  # 后台进程已结束
                taskNameIndex = self.taskNameList.index(self.taskName)
                del self.taskNameList[taskNameIndex]  # 删除对应的任务
                del self.taskProcDict[self.taskName]  # 删除对应的任务进程对象
                self.rds.hdel(f'{self.taskId}_{self.taskName}_train_progress', *
                              self.rds.hkeys(f'{self.taskId}_{self.taskName}_train_progress'))
                # 给Redis发消息,当前训练任务已正常结束
                self.rds.xadd(train_result_topic_name, {
                    'result': f'当前训练任务:{self.taskId}_{self.taskName}正常结束'}, maxlen=100)
            else:
                # 给Redis发消息,出现异常后台进程没有正确结束
                self.rds.xadd(train_result_topic_name, {
                    'result': f'当前训练任务:{self.taskId}_{self.taskName}异常退出'}, maxlen=100)

    def run(self):
        self.generate_dir()
        re_fmt = self.regnix_format()
        self.generate_train_script()
        self.train(re_fmt)
