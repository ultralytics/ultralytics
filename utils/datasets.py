'''
@FileName   :datasets.py
@Description:数据集下载、xml2txt、比例划分、目录移动
@Date       :2025/02/13 15:22:24
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import shutil
import random
import yaml
import asyncio  # 异步包
import xml.etree.ElementTree as ET
from minio import Minio
from minio.error import S3Error
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from .logModels import Logger


class Dataset(Thread):
    '''
    数据集类--继承Thraed,功能点如下：
    支持下载、格式转换、比例划分、路径转移、生成data_yaml文件及数据内容
    '''
    # python类私有属性,防止外部直接修改(但是应该提供访问接口)
    # python类没有真正的私有属性,__Dataset.__ext_name访问
    __ext_name = ('.jpg', '.jpeg', '.png', '.xml')

    def __init__(self, minio: Minio, datasets_dir: str,  bucket: str, prefix: list, labels: list, ratio: float, data_yaml_path: str, log: Logger):
        '''
        初始化线程对象
        params:
            datasets_dir:   训练服务器下载数据集存放目录(固定的)
            bucket:         minio上的桶名(固定的)
            prefix:         数据集名
            labels:         数据集标签列表
            ratio:          数据集划分比例
            data_yaml_path: 训练数据集配置文件路径
            log:日志对象
        '''
        super().__init__()
        self.datasets_dir = datasets_dir
        self.bucket = bucket
        self.prefix_1 = datasets_dir  # minio第一级
        self.prefix_2 = prefix
        self.src_dir = [os.path.join(self.prefix_1, item)
                        for item in self.prefix_2]  # 源目录
        self.labels = labels
        self.ratio = ratio
        self.data_yaml_path = data_yaml_path
        self.log = log
        # [datasets/data1,datasets/data2,...]
        self.dst_dir = [os.path.join(self.datasets_dir, item)
                        for item in prefix]  # 本地目录
        self.minio_client = minio
        # 数据集没下载完成-处理完成-不允许后台训练进程启动
        self.downlaod_finish = False
        # 创建线程池--异步执行各种任务
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.daemon = True  # 外部使用对象.setDaemon()

    # 私有属性访问接口
    @property
    def get_ext_name(self):
        '''
        只读属性,通过@property装饰器实现
        外部只能通过Dataset.get_ext_name获取值,而不能直接修改它
        '''
        return Dataset.__ext_name

    @classmethod  # 静态方法
    def _set_ext_name(cls, new_ext_name):
        '''
        内部使用的方法,允许在一定控制下修改类属性
        注意:该方法前面添加了单下划线,表示不建议外部直接调用
        正确做法:设置setattr、getattr
        '''
        cls.__ext_name = new_ext_name

    def make_local_dir(self):
        ret = []
        try:
            for idx, value in enumerate(self.dst_dir):
                dst_train_dir = os.path.join(value, 'train')
                dst_val_dir = os.path.join(value, 'val')
                # 第一次下载时：
                # 不是第一次下载了,可能是2,3,4...
                if os.path.exists(dst_train_dir) and os.path.exists(dst_val_dir):
                    ret.append(True)
                else:
                    # 创建子目录时,父目录也被创建
                    os.makedirs(dst_train_dir, exist_ok=True)
                    os.makedirs(dst_val_dir, exist_ok=True)
                    ret.append(False)
        except Exception as ex:
            self.log.logger.error(f'创建第{idx}本地目录时出错:{ex}')
        return ret

    def run(self):
        '''
        python中继承Thread必须要重写run()方法,然后调用start()时,类对象自动调用run()
        '''
        # 获取当前线程(Dataset)的事件循环,不存在则新建一个事件循环
        try:
            asyncio.run(self.download_datasets_and_process_datasets())
        except Exception as ex:
            self.log.logger.error(f'Dataset对象线程运行中出错！错误内容:{ex}')

    async def download_files(self, obj, idx):
        '''
        异步下载单个文件,利用run_in_executor()把同步下载任务放入线程池中执行
        param: 
            obj:Minio对象
            idx:prefix索引
        return:
            下载后的本地文件路径
        '''
        local_file_path = os.path.join(
            self.datasets_dir, self.prefix_2[idx], os.path.basename(obj.object_name))
        loop = asyncio.get_event_loop()
        self.log.logger.info(
            f'开始下载{obj.object_name}到{local_file_path}')
        try:
            # 在线程池中调用同步的fget_object()方法实现异步下载
            await loop.run_in_executor(self.executor, self.minio_client.fget_object, self.bucket, obj.object_name, local_file_path)
        except S3Error as s3e:
            self.log.logger.error(f'下载{obj.object_name}失败,错误信息:{s3e}')
        return local_file_path

    async def download_datasets(self):
        repeat_download = self.make_local_dir()  # True:重复下载 False:首次下载--返回[]
        # 重复下载时--不下载数据
        try:
            for idx, value in enumerate(repeat_download):
                if value:
                    self.log.logger.info(f'数据集{self.prefix_2[idx]}已存在,请勿重复选择.')
                else:
                    minio_obj = self.minio_client.list_objects(
                        bucket_name=self.bucket, prefix=os.path.join(self.prefix_1, self.prefix_2[idx]), recursive=True)
                    task = []
                    # 筛选图片和xml文件
                    for obj in minio_obj:
                        if obj.object_name.lower().endswith(Dataset.__ext_name):
                            task.append(self.download_files(obj, idx))
                    if task:
                        # 并发执行所有下载任务
                        await asyncio.gather(*task)
                    else:
                        self.log.logger.info(
                            f'数据集{self.prefix_2[idx]}没有找到符合条件的图片和标签.')
                    task.clear()
        except Exception as ex:
            self.log.logger.error(f'数据集{self.prefix_2[idx]}下载出错,错误内容:{ex}')

    def convert_xml2txt(self, xml_path: str, txt_path: str):
        '''
        将单个xml文件转换为txt格式
        xml文件符合Pascal VOC格式,转换后txt文件的格式为：
        每行"类别索引 xmin ymin xmax ymax"

        :param xml_path: 源xml文件路径
        :param txt_path: 目标txt文件路径
        '''
        lines = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            difficult = 0
            for obj in root.iter('object'):
                if obj.find('difficult'):
                    difficult = obj.find('difficult').text

                # 解析标签名称
                name = obj.find('name').text
                if name not in self.labels or int(difficult) == 1:
                    self.log.logger.info(
                        f'xml标签转换失败,失败原因:name={name}或者difficult={difficult}')
                    return
                else:
                    # 解析边界框坐标
                    bndbox = obj.find('bndbox')
                    bbox = (
                        int(bndbox.find('xmin').text),
                        int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text),
                        int(bndbox.find('ymax').text))
                    bb = self.corrdinate_convert((w, h), bbox)

                    line = f'{self.labels.index(name)} ' + \
                        ' '.join([str(item) for item in bb])
                    lines.append(line)
            # 写入txt
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                f.close()
            self.log.logger.info(f'xml标签转换txt成功:{xml_path}->{txt_path}')
        except Exception as ex:
            self.log.logger.error(f'{xml_path}转换出错:{ex}')

    def corrdinate_convert(self, size: tuple, bbox: tuple):
        '''
        size:(w,h)
        bbox:(xmin,ymin,xmax,ymax)
        '''
        dw = 1./size[0]
        dh = 1./size[1]
        c_x = (bbox[0]+bbox[2])/2.0
        c_y = (bbox[1]+bbox[3])/2.0
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        c_x *= dw
        c_y *= dh
        w *= dw
        h *= dh
        return (c_x, c_y, w, h)

    def _process_sample(self, xml_file: str, dst_dir: str):
        '''
        处理单个样本：
          1. 根据xml文件名查找同目录下的对应图片文件
          2. 移动xml文件和图片文件到目标目录,遇到重名文件则自动重命名
          3. 将xml文件转换为txt格式保存到目标目录
        :param xml_file: 待处理的xml文件路径
        :param dst_dir: 目标目录(train或val)
        '''
        # 获取文件所在目录和基础文件名(不含扩展)
        src_dir = os.path.dirname(xml_file)
        base_name = os.path.splitext(os.path.basename(xml_file))[0]  # 1--对应扩展名
        # 在同级目录查找同名img(支持同名不同扩展的查找,默认第一个)
        img_file = None
        for ext in Dataset.__ext_name:
            candidate = os.path.join(src_dir, base_name+ext)
            if os.path.exists(candidate):
                img_file = candidate
                break
        if not img_file:
            # 不应该跳过--训练时img可以比xml多--这里只考虑完全匹配情况，删除多余的xml
            self.log.logger.info(f'未找到与{xml_file}匹配的图片,跳过该样本.')
            os.remove(xml_file)
            return
        # 移动xml和img文件到目标目录
        xml_path = os.path.join(dst_dir, base_name+'.xml')
        txt_path = os.path.join(dst_dir, base_name+'.txt')
        if os.path.exists(xml_path):
            self.convert_xml2txt(xml_path, txt_path)
        else:
            shutil.move(xml_file, dst_dir)
            shutil.move(img_file, dst_dir)
            # 在目标目录下生成txt--此时xml源目录已变化
            self.convert_xml2txt(xml_path, txt_path)

    def process_datasets(self):
        '''
        处理数据:把下载到本地数据按比例划分;再转换标签格式
          1. 遍历dst_dir下(排除train与val目录)的所有 xml文件,并根据xml文件名查找对应的图片文件(图片后缀匹配)
          2. 按比例将对应的img和xml移动到train或val目录中
          3. 将xml标签转换为txt格式保存(例如转换为"类别 xmin ymin xmax ymax"格式)
        '''
        # 遍历每个数据集的本地存储目录,排除train与val目录
        for idx, value in enumerate(self.dst_dir):
            xml_files = []
            for root, dirs, files in os.walk(value):
                # 排除train和val
                if root.startswith(os.path.join(value, 'train')) or root.startswith(os.path.join(value, 'val')):
                    continue
                for file in files:
                    if file.lower().endswith('.xml'):
                        xml_files.append(os.path.join(root, file))
            if not xml_files:
                self.log.logger.info(f'数据集{value}没有xml文件,无需划分和转换')
                continue
            # 打乱当前数据集xml文件顺序
            random.shuffle(xml_files)
            total = len(xml_files)
            train_count = int(total*self.ratio)
            train_xmls = xml_files[:train_count]
            val_xmls = xml_files[train_count:]
            self.log.logger.info(
                f'{value}目录下共有{total}个xml文件,训练集:{len(train_xmls)}个,验证集:{len(val_xmls)}个')
            # 处理每个xml文件,查找对应图片并移动到目标目录并生成对应的txt标签
            for xml_file in train_xmls:
                self._process_sample(xml_file, os.path.join(value, 'train'))
            for xml_file in val_xmls:
                self._process_sample(xml_file, os.path.join(value, 'val'))

    def generate_yaml(self):
        data_yaml_config = {}
        data_yaml_config['train'] = [os.path.join(
            '../', item, 'train') for item in self.dst_dir]
        data_yaml_config['val'] = [os.path.join(
            '../', item, 'val') for item in self.dst_dir]
        data_yaml_config['names'] = self.labels

        with open(self.data_yaml_path, mode='w', encoding='utf-8') as file:
            # 使用 yaml.dump 将数据写入文件
            # allow_unicode=True 允许写入非 ASCII 字符；sort_keys=False 保持字典顺序
            yaml.dump(data_yaml_config, file,
                      allow_unicode=True, sort_keys=False)
            file.close()

    async def download_datasets_and_process_datasets(self):

        # 异步任务
        await self.download_datasets()
        # 下载完成后,再处理数据集
        self.process_datasets()
        # 然后生成对应的训练数据准备文件yaml
        self.generate_yaml()
