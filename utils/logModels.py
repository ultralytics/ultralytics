'''
@FileName   :logModels.py
@Description:日志记录模块封装
@Date       :2025/02/08 15:03:46
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import os
import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }   # 日志级别

    def __init__(self, filename, level='info', when='midnight', interval=1, backupCount=5, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        '''
            filename: 日志文件名
            level: 日志级别，有debug、info、warning、error、crit五种级别。等级按照顺序递增，高等级日志不会保存低等级日志内容
            when: 时间间隔的时间单位，单位有以下几种：S 秒、 M 分、 H 小时、 D 天、 W 每星期（interval==0时代表星期一）、 midnight 每天凌晨
            interval: 时间间隔，用于确定隔多长时间重新生成一个日志文件。
            backCount: 备份文件的个数，如果超过这个个数，就会自动删除多的日志文件。
            fmt: 日志格式
        '''
        log_path = os.path.dirname(os.path.abspath(filename))
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        self.logger = logging.getLogger(filename)  # 实例化logger
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 输出到屏幕
        sh.setFormatter(format_str)
        # 写入文件
        th = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, interval=interval, backupCount=backupCount, encoding='utf-8')
        th.setFormatter(format_str)
        if not self.logger.handlers:
            # 把输出流对象加载到日志对象中
            self.logger.addHandler(sh)
            self.logger.addHandler(th)
