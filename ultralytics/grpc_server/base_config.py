# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-08-18 16:34:04
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-08-18 21:53:52
@FilePath: /ultralytics/ultralytics/grpc_server/base_config.py
@Description:
'''
import json
import yaml

grpc_param = dict(

detector_params = dict(
    ckpt_path="/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/best.pth",
    thr={'default':0.3},
    change_label=dict(wcgz_dxdk="wcgz"),
    device="cuda:0",
    nms=0.5,
),

grpc_args = dict(
    host="127.0.0.1",
    port="7999",
    max_workers=1,
    max_send_message_length=512 * 1024 * 1024,
    max_receive_message_length=512 * 1024 * 1024,
)

)

def _update(dic1: dict, dic2: dict):
    """使用dic2 来递归更新 dic1
        # NOTE:
        1. dic1 本体是会被更改的!!!
        2. python 本身没有做尾递归优化的,dict深度超大时候可能爆栈
    """
    for k, v in dic2.items():
        if k.endswith('args') and v is None:
            dic2[k] = {}
        if k in dic1:
            if isinstance(v, dict) and isinstance(dic1[k], dict):
                _update(dic1[k], dic2[k])
            else:
                dic1[k] = dic2[k]
        else:
            dic1[k] = dic2[k]

def _merge_yaml(yaml_path: str):
    global grpc_param
    with open(yaml_path, 'r') as fr:
        content_dict = yaml.load(fr, yaml.FullLoader)
    _update(grpc_param, content_dict)

def _merge_json(json_path: str):
    global grpc_param
    with open(json_path, 'r') as fr:
        content_dict = json.load(fr)
    _update(grpc_param, content_dict)

def merge_param(file_path: str):
    """按照用户传入的配置文件更新基本设置
    """
    cfg_ext = file_path.split('.')[-1]
    func_name = '_merge_' + cfg_ext
    if func_name not in globals():
        raise ValueError('{} is not support'.format(cfg_ext))
    else:
        globals()[func_name](file_path)
