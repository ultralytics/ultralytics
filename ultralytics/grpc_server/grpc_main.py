# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-08-18 16:28:47
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-08-18 19:35:13
@FilePath: /ultralytics/ultralytics/grpc_server/grpc_main.py
@Description:
'''
import os
import argparse
from concurrent import futures
from pprint import pprint
from datetime import datetime
import asyncio

import grpc
from .proto import dldetection_pb2_grpc as dld_grpc

from .model import Detector
from . import base_config as config_manager



async def main(cfg_path):
    if os.path.exists(cfg_path):
        config_manager.merge_param(cfg_path)
    args_dict: dict = config_manager.grpc_param
    print("current time is: ", datetime.now())

    pprint(args_dict)

    grpc_args = args_dict['grpc_args']
    detector_params = args_dict['detector_params']
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=grpc_args['max_workers']),
        options=[('grpc.max_send_message_length',
                  grpc_args['max_send_message_length']),
                 ('grpc.max_receive_message_length',
                  grpc_args['max_receive_message_length'])])
    model = Detector(**detector_params)
    dld_grpc.add_AiServiceServicer_to_server(model, server)

    server.add_insecure_port("{}:{}".format(grpc_args['host'],
                                            grpc_args['port']))
    await server.start()
    print('yolo gprc server init done')
    await server.wait_for_termination()

def run_grpc(cfg_path):
    asyncio.run(main(cfg_path))