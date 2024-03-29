import yaml
import inspect
import logging
import torch.nn as nn
import torch.optim as opt
from logging import handlers
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler


class Identity(nn.Module):
    """
    Identity mapping.
    This is generally used as "Identity activation" in networks for
    convenient implementation of "no activation".
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def load_yaml_file(file_path):
    with open(file_path, mode="r", encoding="utf-8") as file:
        data = file.read()

    data = yaml.load(data, Loader=yaml.FullLoader)
    return data


def get_transformations(argumentation_configs, module=transforms):
    trans = []
    trans_available = {}
    for name, method in inspect.getmembers(module):
        if name[0] != "_":
            trans_available[name] = method

    for key in argumentation_configs:
        if key in trans_available.keys():
            trans.append(trans_available[key](**argumentation_configs[key]))
    return trans


def get_scheduler(optimizer, scheduler_configs):
    schedulers = []
    schedulers_available = {}
    for name, scheduler in inspect.getmembers(lr_scheduler):
        if name[0] != "_":
            schedulers_available[name] = scheduler

    for key in scheduler_configs.keys():
        schedulers.append(schedulers_available[key](optimizer, **scheduler_configs[key]))
    if len(schedulers) == 0:
        raise NotImplementedError
    schedulers = lr_scheduler.ChainedScheduler(schedulers)
    return schedulers


def get_optimizer(params, optimizer_configs):
    optimizers_available = {}
    for name, optimizer in inspect.getmembers(opt):
        if name[0] != "_":
            optimizers_available[name] = optimizer

    optimizer_name = list(optimizer_configs.keys())
    if len(optimizer_name) > 1:
        raise Exception("Not support more than one optimizers!")
    if len(optimizer_name) == 0:
        raise Exception("At least one optimizer must be specified!")
    optimizer_name = optimizer_name[0]
    args = optimizer_configs[optimizer_name]
    for key, value in args.items():
        try:
            args[key] = float(value)
        except Exception:
            pass
    return optimizers_available[optimizer_name](params=params, **args)


def init_nn(model, init_configs):
    inits_available = {}
    for name, init in inspect.getmembers(nn.init):
        # All init methods are end with "_"
        if name[-1] == "_":
            inits_available[name] = init

    init_name = list(init_configs.keys())
    if len(init_name) > 1:
        raise Exception("Not support more than one init methods!")
    if len(init_name) == 0:
        raise Exception("At least one init method must be specified!")
    init_name = init_name[0]
    args = init_configs[init_name]
    for key, value in args.items():
        try:
            args[key] = float(value)
        except Exception:
            pass
    for layer, name in model.named_modules():
        if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
            inits_available[init_name](layer.weight, **args)


def cal_model_complexity(model, input_shape, log):
    macs, params = get_model_complexity_info(model,
                                             input_shape,
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    log.logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    log.logger.info('{:<30}  {:<8}'.format('Number of parameters    : ', params))


def parse_device(device_name):
    """
    The simplest way to set the value is:
      1) set to "cpu" if cpu is used;
      2) set to a list of IDs (int) if GPUs are used.
         Data will be collected on the first GPU in the list.
      Examples:
        1) Set to [0] if only one GPU is used.
        2) Set to [1, 0] if two GPUs are used. Data will be collected on GPU 1.
    """
    device_id, device_ids, device = None, None, None
    if isinstance(device_name, str):
        assert device_name in ["cuda", "cpu"]
        if device_name == "cuda":
            device_id = "cuda:0"
            device_ids = [0]
            device = "cuda"
        else:
            device = "cpu"
            device_id = "cpu"
    elif isinstance(device_name, list):
        device_id = f"cuda:{device_name[0]}"
        device_ids = list(device_name)
        device = "cuda"
    return device_id, device_ids, device


def set_device(model, device_id, device_ids, device):
    # Set GPU mode if GPU used
    if device == "cuda":
        model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device_id)
    return model


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }    

    def __init__(self, filename, level='info', fmt="%(asctime)s - %(levelname)s - %(message)s"):
        DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
        logging.basicConfig(datefmt=DATE_FORMAT)
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)                  
        self.logger.setLevel(self.level_relations.get(level))
        
        sh = logging.StreamHandler()  
        sh.setFormatter(format_str)   
        self.logger.addHandler(sh)   
        
        fh = handlers.RotatingFileHandler(filename=filename, mode='w') 
        fh.setLevel(self.level_relations.get(level))
        fh.setFormatter(format_str) 
        self.logger.addHandler(fh)


def draw_line_figure(data_list, figsize, dpi, x_label, y_label, legend_loc, save_path):
    """
    Helper function to draw and save figure.

    Parameters
    ----------
    data_list: list[list]
      A list of data to draw.
      The sublist is assumed to be a 4-sized list:
      1) The 1st element is the values of y-axis;
      2) The 2nd element is the values of x-axis;
      3) The 3rd element is the line color;
      4) The 4th element is the line label name.
    figsize: Tuple(int, int)
      Figure size.
    dpi: int
      Dots per inches (dpi) determines how many pixels the figure comprises.
    x_label: str
      x label name.
    y_label: str
      y label name.
    legend_loc: str
      Location of legend.
    save_path: str
      Path of the figure to save.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    for data in data_list:
        data = list(data)
        plt.plot(data[0], data[1], color=data[2], label=data[3])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_loc)
    plt.savefig(save_path)
    plt.close()


def set_params(default_params, configs, excluded_keys=[]):
    """
    Helper function to set params.
    This is generally used in the model's "make_network" function.
    The method tries to use param's value in configs to replace the corresponding
    value in default_params.

    Parameters
    ----------
    default_params: dict[str: obj]
      A dict of default params with name (key) and value (value).
    configs: dict[str: obj]
      Config.
    excluded_keys: list[str]
      Keys to exclude in configs.
    """
    default_keys = list(default_params.keys())
    config_keys = list(configs.keys())
    for key in config_keys:
        if key in default_keys:
            if key not in excluded_keys:
                default_params[key] = configs[key]
        else:
            raise Exception(f"Error key: {key}!")

    return default_params