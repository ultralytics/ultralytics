"""
@FileName   :exportPaddleTask.py
@Description:
@Date       :2025/02/28 15:14:52
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com.
"""

import datetime
import subprocess
from threading import Thread

import psutil
from redis import Redis

from config import export_result, export_result_topic_name


class exportPaddleTask(Thread):
    """yolo目标检测训练类."""

    def __init__(
        self, rds: Redis, taskId: str, taskName: str, taskNameList: list, taskProcDict: dict, modelId: str, call_back
    ):
        super().__init__()
        self.daemon = True
        self.modelId = modelId  # 是上次模型训练的唯一标识(taskId_taskName)
        self.rds = rds
        self.taskId = taskId
        self.taskName = taskName
        self.taskNameList = taskNameList
        self.taskProcDict = taskProcDict
        self.func_call_back = call_back  # 通用的回调函数属性

    def generate_export_script(self):
        """生成导出脚本."""
        file = open("./export.py", "w", encoding="utf-8")
        file.write("# 此文件由程序自动生成.\n")
        file.write("# 请勿手动修改.\n")
        file.write("from ultralytics import YOLO\n")
        file.write("w = 640\n")
        file.write("h = 640\n")
        file.write(f"model_path = './{export_result}/{self.modelId}.pt'\n")
        file.write("model = YOLO(model_path)\n")
        file.write("model.export(format='paddle',imgsz=[w,h],opset=12,device='0')\n")
        file.close()

    def export(self):
        self.rds.hset(f"{self.taskId}_{self.taskName}_export_progress", mapping={"status": 1, "epoch": 0})
        log_file = open(f"./log/export_log_{self.taskId}_{self.taskName}.txt", "w", encoding="utf-8")
        train_command = ["source activate paddle && python export.py"]
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
            self.func_call_back(
                "start",
                self.taskId,
                self.taskName,
                0,
                f"当前导出任务:{self.taskName}启动失败,后台进程已终止.",
                create_time,
            )
            taskNameIndex = self.taskNameList.index(self.taskName)
            del self.taskNameList[taskNameIndex]  # 删除对应的任务
            self.rds.hdel(
                f"{self.taskId}_{self.taskName}_export_progress",
                *self.rds.hkeys(f"{self.taskId}_{self.taskName}_export_progress"),
            )
            log_file.close()
        else:  # 正常启动
            self.func_call_back(
                "start",
                self.taskId,
                self.taskName,
                1,
                f"当前导出任务:{self.taskId}_{self.taskName}启动成功,后台进程已启动.",
                create_time,
            )
            self.taskProcDict[self.taskName] = process  # self.export_task_obj
            for line in process.stdout:
                log_file.write(line)  # 写入导出日志文档
                log_file.flush()
                print(line)
            log_file.close()
            returncode = process.wait()  # 会一直到等到当前训练结束才返回0--正常结束;1--异常返回
            status = 0 if returncode == 0 else -1
            self.rds.hset(f"{self.taskId}_{self.taskName}_export_progress", mapping={"status": status, "epoch": 0})
            if not psutil.pid_exists(process.pid):  # 后台进程已结束
                taskNameIndex = self.taskNameList.index(self.taskName)
                del self.taskNameList[taskNameIndex]  # 删除对应的任务
                del self.taskProcDict[self.taskName]  # 删除对应的任务进程对象
                self.rds.hdel(
                    f"{self.taskId}_{self.taskName}_export_progress",
                    *self.rds.hkeys(f"{self.taskId}_{self.taskName}_export_progress"),
                )
                # 给Redis发消息,当前训练任务已正常结束
                self.rds.xadd(
                    export_result_topic_name,
                    {"result": f"当前导出任务:{self.taskId}_{self.taskName}正常结束"},
                    maxlen=100,
                )
            else:
                # 给Redis发消息,出现异常后台进程没有正确结束
                self.rds.xadd(
                    export_result_topic_name,
                    {"result": f"当前导出任务:{self.taskId}_{self.taskName}异常退出"},
                    maxlen=100,
                )

    def run(self):
        self.generate_export_script()
        self.export()
