# Ultralytics YOLO 🚀, GPL-3.0 license

import os
import shutil
import socket
import sys
import tempfile

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer):
    import_path = '.'.join(str(trainer.__class__).split(".")[1:-1])

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    content = f'''cfg = {vars(trainer.args)} \nif __name__ == "__main__":
    from ultralytics.{import_path} import {trainer.__class__.__name__}

    trainer = {trainer.__class__.__name__}(cfg=cfg)
    trainer.train()'''
    (USER_CONFIG_DIR / 'DDP').mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="_temp_",
                                     suffix=f"{id(trainer)}.py",
                                     mode="w+",
                                     encoding='utf-8',
                                     dir=USER_CONFIG_DIR / 'DDP',
                                     delete=False) as file:
        file.write(content)
    return file.name


def generate_ddp_command(world_size, trainer):
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
    file = generate_ddp_file(trainer) if sys.argv[0].endswith('yolo') else os.path.abspath(sys.argv[0])
    torch_distributed_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    cmd = [
        sys.executable, "-m", torch_distributed_cmd, "--nproc_per_node", f"{world_size}", "--master_port",
        f"{find_free_network_port()}", file] + sys.argv[1:]
    return cmd, file


def ddp_cleanup(trainer, file):
    # delete temp file if created
    if f"{id(trainer)}.py" in file:  # if temp_file suffix in file
        os.remove(file)
