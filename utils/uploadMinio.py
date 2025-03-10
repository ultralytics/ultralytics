"""
@FileName   :uploadMinio.py
@Description:
@Date       :2025/02/21 16:31:54
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

from minio import Minio
from minio.error import S3Error
from redis import Redis

from .logModels import Logger


class uploadMinio(Thread):
    """通用的文件上传类."""

    def __init__(
        self,
        rds: Redis,
        minio_client: Minio,
        bucket: str,
        result: str,
        taskId: str,
        taskName: str,
        minio_prefix: str,
        uploadType: str,
        modelId,
        log: Logger,
        max_workers: int = 10,
    ):
        """初始化uploadMinio类 :param minio_client:    已初始化的Minio客户端 :param bucket:          上传目标桶名称 :param result:
        本地训练的目录 :param taskId:          训练任务的id :param taskName:        训练任务的name :param minio_prefix:    minio的prefix
        :param uploadType:      minio上传类型,训练结果上传,导出结果上传 :param modelId:         导出时采用到参数,用于找到导出的模型 :param log:
        日志 :param max_workers:     线程池最大工作线程数.
        """
        super().__init__()
        self.rds = rds
        self.minio_client = minio_client
        self.bucket = bucket
        if uploadType == "train":
            self.rds_name = f"{taskId}_{taskName}_train_progress"  # 训练时的消息流名
            # train--本地待上传的文件夹路径
            self.local_folder = f"{result}/{taskId}_{taskName}"

        else:
            self.rds_name = f"{taskId}_{taskName}_export_progress"  # 导出时的消息流名
            # export--本地待上传的文件夹路径
            self.local_folder = f"{result}/{modelId}_paddle_model"
        # 去除末尾的'/',方便后续路径拼接
        # 桶内目标目录前缀(上传后所有文件存放在该目录下)
        self.remote_prefix = f"{minio_prefix}/{taskId}_{taskName}".rstrip("/")
        self.log = log
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _get_all_files(self):
        """遍历本地文件夹及其子目录,返回一个包含所有文件完整路径和相对路径的列表 :return: [(full_path, relative_path), ...]."""
        files_list = []
        for root, _, files in os.walk(self.local_folder):
            for file in files:
                full_path = os.path.join(root, file)
                # 计算相对于local_folder的相对路径
                rel_path = os.path.relpath(full_path, self.local_folder)
                files_list.append((full_path, rel_path))
        return files_list

    async def upload_file(self, full_path: str, rel_path: str):
        """异步上传单个文件到Minio桶中 :param full_path: 本地文件的完整路径 :param rel_path: 文件相对于本地根目录的相对路径."""
        # 构造远程对象名:remote_prefix+'/'+rel_path
        remote_key = f"{self.remote_prefix}/{rel_path}"
        # 把系统路径符转换为'/'--minio统一使用/
        remote_key = remote_key.replace("\\", "/")
        loop = asyncio.get_event_loop()
        self.log.logger.info(f"开始上传:{full_path}->{remote_key}")
        try:
            # 在线程池中调用同步函数fput_object()上传文件
            await loop.run_in_executor(self.executor, self.minio_client.fput_object, self.bucket, remote_key, full_path)
            self.log.logger.info(f"上传成功:{full_path}")
        except S3Error as s3e:
            self.log.logger.error(f"上传文件{full_path}出错,错误:{s3e}")

    async def upload_all_files(self):
        """异步任务:并发上传所有文件."""
        tasks = []
        files = self._get_all_files()
        for full_path, rel_path in files:
            tasks.append(self.upload_file(full_path, rel_path))
        # 并发执行所有上传任务
        await asyncio.gather(*tasks)

    def run(self):
        """重写Thread的run方法,启动asyncio事件循环执行所有上传任务."""
        while True:
            if not self.rds.exists(self.rds_name):
                try:
                    asyncio.run(self.upload_all_files())
                except Exception as ex:
                    self.log.logger.error(f"上传过程发生错误:{ex}")
                break
            else:
                time.sleep(3)
                continue
