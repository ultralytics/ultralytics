import json
import os
import time

import cv2
import numpy as np
import requests
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from concurrent import futures

idx = 0

failed = 0

urls = json.load(open("/home/lixiang/下载/workbook/urls.json"))

MAX_WORKERS = 500

# res = requests.get(image_url)
# img = cv2.imdecode(np.fromstring(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
# 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。
auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
# 填写源Bucket名称，例如srcexamplebucket。
src_bucket_name = 'ow-prod'
# 填写与源Bucket处于同一地域的目标Bucket名称，例如destexamplebucket。
# 当在同一个Bucket内拷贝文件时，请确保源Bucket名称和目标Bucket名称相同。
dest_bucket_name = 'ow-prod'
# yourEndpoint填写Bucket所在Region对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
bucket = oss2.Bucket(auth, 'https://oss-cn-beijing.aliyuncs.com', dest_bucket_name)
url_batches = np.array_split(urls, MAX_WORKERS)


def copy_img(url):
    # 填写不包含Bucket名称在内源Object的完整路径，例如srcexampleobject.txt。
    src_object_name = url[1:]
    # 填写不包含Bucket名称在内目标Object的完整路径，例如destexampleobject.txt。
    dest_object_name = os.path.join('美丽海淀图片', url.split('/')[-1])
    # 将源Bucket中的某个Object拷贝到目标Bucket。
    result = bucket.copy_object(src_bucket_name, src_object_name, dest_object_name)
    # 查看返回结果的状态。如果返回值为200，表示执行成功。
    # img = cv2.imread(image_url)
    if result.status != 200:
        print(result.resp)

idx = 0
begin = time.time()
for urls in url_batches:
    batch_begin = time.time()
    with futures.ThreadPoolExecutor(max_workers=max(MAX_WORKERS, len(urls))) as executor:  # 实例化线程池
        res = executor.map(copy_img, urls)

        # 跟内置的map很像，对序列进行相同操作，注意是异步、非阻塞的！
        # 返回的是一个生成器，需要调用next
    idx += len(urls)
    print(f'已复制{idx}张图片，当前批次耗时{time.time() - batch_begin}秒，总耗时{time.time() - begin}秒')
    # img = cv2.imread(image_url)
    # cv2.imwrite(os.path.join("/home/lixiang/下载/workbook/图片", image_url.split("/")[-1]), img)

