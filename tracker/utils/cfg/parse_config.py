import json
import torch
from collections import OrderedDict
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime


class ConfigParser:
    def __init__(self, config, modification=None):
        """
        class to parse configuration json file. Handles hyperparameters for
        training, initializations of modules, checkpoint saving and
        logging module.
        :param config: Dict containing configurations, hyperparameters for
                       training. contents of `config.json` file for example.
        :param modification: Dict keychain:value, specifying position values
                             to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save
                       checkpoints and training log. Timestamp is being used
                       as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)

        # set save_dir where outputs will be saved.
        save_dir = Path(self.config["save_dir"])

        exper_name = self.config["name"] + '/' + datetime.now().strftime(r"%m%d_%H%M%S")
        self._save_dir = save_dir / exper_name

        # make directory for saving checkpoints and log.
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / "config.json")

    @classmethod
    def from_args(cls, args):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if not isinstance(args, tuple):
            args = args.parse_args()

        msg_no_cfg = "Configuration file need to be specified. Add \
            '-c config.json', for example."
        assert args.config is not None, msg_no_cfg
        resume = None
        cfg_fname = Path(args.config)

        config = read_json(cfg_fname)

        return cls(config, resume)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and
        returns the instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and
        returns the function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(
            a, *args, b=1, **kwargs
            )`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(";")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)

# Json utils
def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices
    which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use},\
                  but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids