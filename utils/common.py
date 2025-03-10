import datetime
import os
import re
import subprocess

import psutil
from redis import Redis

from config import data_cfg, pretrained_models, train_result, train_result_topic_name


def file_save(fileDir: str, fileName: str, file):
    if not os.path.exists(fileDir):
        os.makedirs(fileDir, exist_ok=True)
    filePath = os.path.join(fileDir, fileName)
    try:
        with open(filePath, "wb") as f:
            while True:
                chunk = file.read(1024)
                if not chunk:  # 不为空
                    break
                f.write(chunk)
    except Exception as ex:
        print(f"exception:{ex}")


def handle_pipe(pipe, epoch_pattern, epoch, log_file):
    if pipe is None:
        print("PIPE is null.......")
    else:
        for line in iter(pipe.readline, ""):
            ln = line.strip()

            match = epoch_pattern.search(ln)
            if match:
                current_epoch = int(match.group(1))
                total_epochs = int(match.group(2))
                epoch = f"Current Epoch:{current_epoch}/{total_epochs}"
                print(epoch)
                log_file.write(ln)


def find_pt(src_dir, target_name: str):
    """
    功能:遍历指定目录及其子目录,查找目标.pt文件.

    参数:   src_dir:需要遍历的根目录路径。   target_name:要查找的.pt文件的名称,可以带或不带.pt后缀

    返回:   如果找到目标文件,返回该文件的完整路径;如果未找到,则返回 None。
    """
    if not target_name.endswith(".pt"):
        target_name = "".join([target_name, ".pt"])
    # 使用os.walk遍历目录及子目录
    for root, _, files in os.walk(src_dir):
        for file in files:
            # 检查文件是否和目标匹配
            if file.endswith(target_name):
                return os.path.join(root, file)
    return None  # 没有匹配到,返回None


def train_yolo_classify_task(rds, taskName, taskNameList, taskProcDict, task_id, parameters):
    """分类模型训练函数."""
    pass


# 此方法已弃用
def train_yolo_detect_task(
    rds: Redis,
    taskId: str,
    taskName: str,
    taskNameList: list,
    taskProcDict: dict,
    labels: list,
    netType: str,
    trainType: str,
    modelId: str,
    parameters: dict,
    call_back,
):
    """
    检测模型训练函数
    rds:            redis对象
    taskId:         用于标识任务的唯一ID
    taskName:       当前训练任务名
    taskNameList:   所有训练任务列表
    taskProcDict:   所有训练任务后台进程字典
    parameters：    训练参数(目前作为占位--字典)
    call_back:      外部回调函数.
    """
    if not os.path.exists(data_cfg):
        os.makedirs(data_cfg)
    elif not os.path.exists(train_result):
        os.makedirs(train_result)
    elif not os.path.exists(pretrained_models):
        os.makedirs(pretrained_models)

    train_params = {
        "data": f"{data_cfg}/{taskId}_{taskName}.yaml",
        "project": f"{train_result}",
        "name": f"{taskId}_{taskName}",
        "task": "detect",
    }
    re_list = []
    r1 = r"(\d+)/"
    re_list.append(f"{r1}{parameters['epochs']}")
    r1 = r"^      Epoch"
    re_list.append(f"{r1}")
    r1 = r"^                 Class"
    re_list.append(f"{r1}")
    r1 = r"^                   all"
    re_list.append(f"{r1}")
    for i in range(len(labels)):
        re_list.append(f"^                   {labels[i]}")
    re_format = "|".join(re_list)
    print(f"re_format:{re_format}")
    with open("./train.py", "w", encoding="utf-8") as file:
        file.write("# 此文件由程序自动生成.\n")
        file.write("# 请勿手动修改.\n")

        # 开始写入训练代码
        file.write("import os\n")
        file.write("import shutil\n")
        file.write("import datetime\n")
        file.write("from ultralytics import YOLO\n")
        if trainType == "Init":  # 初始化训练
            if netType.startswith("yolov5"):
                file.write(f"model_cfg = '{pretrained_models}/{netType}u.yaml'\n")
                file.write(f"model_path = '{pretrained_models}/{netType}u.pt'\n")
            else:
                file.write(f"model_cfg = '{pretrained_models}/{netType}.yaml'\n")
                file.write(f"model_path = '{pretrained_models}/{netType}.pt'\n")
        else:  # 迭代--modelId是上一次训练的taskId_taskName
            iter_pre_model = find_pt("models", modelId)
            if netType.startswith("yolov5"):
                file.write(f"model_cfg = '{pretrained_models}/{netType}u.yaml'\n")
                file.write(f"model_path = '{iter_pre_model}'\n")
            else:
                file.write(f"model_cfg = '{pretrained_models}/{netType}.yaml'\n")
                file.write(f"model_path = '{iter_pre_model}'\n")

        for key, value in parameters.items():
            # 使用repr()确保value被合法转换
            # file.write(f"{key} = {repr(value)}\n")
            # file.write(f"{key} = {value}\n")
            train_params[key] = value
        file.write(f"train_params={train_params}\n")
        file.write("model = YOLO(model_cfg).load(model_path)\n")
        file.write("model.train(**train_params)\n")
        # 训练完成后的，模型移动操作代码块
        model_trained_path = "/".join([train_params["project"], train_params["name"], "weights", "best.pt"])
        file.write(f"model_trained_path = '{model_trained_path}'\n")
        file.write(f"if not os.path.exists('{'models'}'):\n")
        file.write(f"    os.makedirs('{'models'}')\n")
        model_rename = f"datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_{taskId}_{taskName}'"
        file.write(f"model_rename = {model_rename}\n")
        file.write("model_rename = ''.join([model_rename,'.pt'])\n")
        file.write("model_rename_path = '/'.join(['models',model_rename])\n")
        file.write("shutil.copy(model_trained_path,model_rename_path)\n")
        file.close()

    rds.hset(f"{taskId}_{taskName}_train_progress", mapping={"status": 1, "epoch": 0})
    log_file = open(f"./log/train_log_{taskId}_{taskName}.txt", "w", encoding="utf-8")
    # Logger()--start_train_taskId_taskName.txt--记录进程、训练起始等中间信息(用于debug)*******
    # train_args = shlex.split(train_command)
    train_command = ["source activate paddle && python train.py"]
    process = subprocess.Popen(
        args=train_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=True,
        executable="/bin/bash",
    )
    # 回调函数
    create_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if not psutil.pid_exists(process.pid):  # 后台程序出错
        call_back("start", taskId, taskName, 0, f"当前训练任务:{taskName}启动失败,后台进程已终止.", create_time)
        taskNameIndex = taskNameList.index(taskName)
        del taskNameList[taskNameIndex]  # 删除对应的任务
        rds.hdel(f"{taskId}_{taskName}_train_progress", *rds.hkeys(f"{taskId}_{taskName}_train_progress"))
        log_file.close()
    else:  # 正常启动
        call_back(
            "start", taskId, taskName, 1, f"当前训练任务:{taskId}_{taskName}启动成功,后台进程已启动.", create_time
        )
        taskProcDict[taskName] = process  # self.train_task_obj
        for line in process.stdout:
            log_file.write(line)  # 这里看需求写入文档,目前只传epoch p r mAP数据传送到Redis
            log_file.flush()
            match_ret = re.search(re_format, line)
            # print("match:\t", match_ret)
            if match_ret:
                if not match_ret.group(1):
                    print(line)
                else:
                    print(line)
                    current_epoch = int(match_ret.group(1))
                    # 通过Redis发送中间训练过程
                    rds.hset(f"{taskId}_{taskName}_train_progress", mapping={"status": 1, "epoch": current_epoch})
                    # 可能还会有其他训练过程信息发送....
        log_file.close()
        returncode = process.wait()  # 会一直到等到当前训练结束才返回0--正常结束;1--异常返回
        status = 0 if returncode == 0 else -1
        rds.hset(f"{taskId}_{taskName}_train_progress", mapping={"status": status, "epoch": 0})
        if not psutil.pid_exists(process.pid):  # 后台进程已结束
            taskNameIndex = taskNameList.index(taskName)
            del taskNameList[taskNameIndex]  # 删除对应的任务
            del taskProcDict[taskName]  # 删除对应的任务进程对象
            rds.hdel(f"{taskId}_{taskName}_train_progress", *rds.hkeys(f"{taskId}_{taskName}_train_progress"))
            # 给Redis发消息,当前训练任务已正常结束
            rds.xadd(train_result_topic_name, {"result": f"当前训练任务:{taskId}_{taskName}正常结束"}, maxlen=100)
        else:
            # 给Redis发消息,出现异常后台进程没有正确结束
            rds.xadd(train_result_topic_name, {"result": f"当前训练任务:{taskId}_{taskName}异常退出"}, maxlen=100)


def train_yolo_segment_task(rds, taskName, taskNameList, taskProcDict, task_id, parameters):
    """分割模型训练函数."""
    pass


def train_yolo_pose_task(rds, taskName, taskNameList, taskProcDict, task_id, parameters):
    """姿态估计模型训练函数."""
    pass


def train_yolo_obb_task(rds, taskName, taskNameList, taskProcDict, task_id, parameters):
    """旋转检测模型训练函数."""
    pass


def train_yolo_track_task(rds, taskName, taskNameList, taskProcDict, task_id, parameters):
    """跟踪模型训练函数."""
    pass
