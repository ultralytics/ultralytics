# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A script to run multinode training with submitit.
Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
"""
import argparse
import os
import uuid
from pathlib import Path

import submitit

from . import main_dino


def parse_args():
    parser = argparse.ArgumentParser('Submitit for DINO', parents=[main_dino.get_args_parser()])
    parser.add_argument('--ngpus', default=8, type=int, help='Number of gpus to request on each node')
    parser.add_argument('--nodes', default=2, type=int, help='Number of nodes to request')
    parser.add_argument('--timeout', default=2800, type=int, help='Duration of the job')

    parser.add_argument('--partition', default='learnfair', type=str, help='Partition where to submit')
    parser.add_argument('--use_volta32', action='store_true', help='Big models? Use this')
    parser.add_argument('--comment', default='', type=str, help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv('USER')
    if Path('/checkpoint/').is_dir():
        p = Path(f'/checkpoint/{user}/experiments')
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError('No shared folder available')


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f'{uuid.uuid4().hex}_init'
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer:

    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main_dino

        self._setup_gpu_args()
        main_dino.train_dino(self.args)

    def checkpoint(self):
        import os

        import submitit

        self.args.dist_url = get_init_file().as_uri()
        print('Requeuing ', self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path

        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace('%j', str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f'Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}')


def main():
    args = parse_args()
    if args.output_dir == '':
        args.output_dir = get_shared_folder() / '%j'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs)

    executor.update_parameters(name='dino')

    args.dist_url = get_init_file().as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f'Submitted job_id: {job.job_id}')
    print(f'Logs and checkpoints will be saved at: {args.output_dir}')


if __name__ == '__main__':
    main()
