import yaml
import inspect
import logging
import torch.nn as nn
import torch.optim as opt
from logging import handlers
from functools import partial
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


class IdentityNorm(nn.Module):
    """
    Identity norm.
    This is generally used as "Identity norm" in networks for
    convenient implementation of "no normalization".
    """

    def __init__(self, in_channels):
        super(IdentityNorm, self).__init__()

    def forward(self, x):
        return x


def load_yaml_file(file_path):
    with open(file_path, mode="r", encoding="utf-8") as file:
        data = file.read()

    data = yaml.load(data, Loader=yaml.FullLoader)
    return data


NORMS = {
    "BatchNorm": nn.BatchNorm2d,
    "IdentityNorm": IdentityNorm,
}


def get_norm(norm_type):
    def get_norm_by_name(name):
        if name in NORMS:
            return NORMS[name]
        else:
            raise NotImplementedError

    if isinstance(norm_type, str):
        norm = get_norm_by_name(norm_type)
    else:
        # "norm_type" is a dict
        norm_name = list(norm_type.keys())
        if len(norm_name) > 1:
            raise Exception("Not support more than one norm method!")
        if len(norm_name) == 0:
            raise Exception("At least one norm method must be specified!")
        norm_name = norm_name[0]
        norm = get_norm_by_name(norm_name)
        norm_configs = norm_type[norm_name]
        norm = partial(norm, **norm_configs)
    return norm


ACTS = {
    "Identity": Identity,
    "Relu": nn.ReLU,
    "ReLU": nn.ReLU,
    "LeakyRelu": nn.LeakyReLU,
    "LeakyReLU": nn.LeakyReLU,
}


def get_activation(act_type):
    def get_act_by_name(name):
        if name in ACTS:
            return ACTS[name]
        else:
            raise NotImplementedError

    if isinstance(act_type, str):
        act = get_act_by_name(act_type)
    else:
        # "act_type" is a dict
        act_name = list(act_type.keys())
        if len(act_name) > 1:
            raise Exception("Not support more than one activation function!")
        if len(act_name) == 0:
            raise Exception("At least one activation function must be specified!")
        act_name = act_name[0]
        act = get_act_by_name(act_name)
        act_configs = act_type[act_name]
        act = partial(act, **act_configs)
    return act


def get_transformations(argumentation_configs):
    trans = []
    trans_available = {}
    for name, method in inspect.getmembers(transforms):
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
        if name[-1] != "_":
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
            inits_available[init_name](layer.weightm, **args)


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
