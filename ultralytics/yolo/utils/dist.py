import os
import socket
import sys


def find_free_network_port() -> int:
    # https://github.com/Lightning-AI/lightning/blob/master/src/lightning_lite/plugins/environments/lightning.py
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def generate_ddp_command(world_size):
    import __main__  # local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

    return [
        sys.executable, "-m", "torch.distributed.launch", "--nproc_per_node", f"{world_size}", "--master_port",
        f"{find_free_network_port()}",
        os.path.abspath(sys.argv[0])] + sys.argv[1:]
