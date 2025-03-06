'''
@FileName   :config.py
@Description:
@Date       :2025/02/10 11:30:55
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
# Redis连接相关配置
redis_ip = "192.168.1.184"
redis_port = 6379
redis_pwd = ""
# 训练服务器IP
IP = "192.168.1.184"
# 数据集配置
minio_endpoint = "192.168.1.134:9000"
minio_access_key = "minioadmin"
minio_secret_key = "minioadmin"
# 训练的Redis消息流相关配置
train_host_msg = "AI_TRAIN_HOST_MSG"  # 暂时没用到
train_action_opt_topic_name = "AI_TRAIN_TASK_ACTION_OPT"  # 训练任务下发的消息流
train_action_result_topic_name = "AI_TRAIN_TASK_ACTION_RESULT"  # 训练任务返回的消息流
train_data_download_topic_name = "AI_TRAIN_DATA_DOWNLOAD_RESULT"  # 训练数据下载的消息流
train_result_topic_name = "AI_TRAIN_TASK_RESULT"  # 训练任务的完成情况消息流
# 导出的Redis消息流相关配置
export_host_msg = "AI_EXPORT_HOST_MSG"
export_action_opt_topic_name = "AI_EXPORT_TASK_ACTION_OPT"  # 导出任务下发的消息流
export_action_result_topic_name = "AI_EXPORT_TASK_ACTION_RESULT"  # 导出任务返回的消息流
export_result_topic_name = "AI_EXPORT_TASK_RESULT"  # 导出结果的消息流
# 数据集路径配置目录
data_cfg = 'data_cfg'
# 预训练模型目录
pretrained_models = 'pretrained_models'
# 训练结果目录
train_result = 'train_results'
# 上传minio的prefix
minio_train_prefix = 'train_result_package'
minio_export_prefix = 'export_result_package'
# 导出结果目录
export_result = 'export_results'
