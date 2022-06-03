import yaml
import logging
import torch.nn as nn
from logging import handlers
import torchvision.transforms as transforms

import resnet


def load_yaml_file(file_path):
    with open(file_path, mode="r", encoding="utf-8") as file:
        data = file.read()

    data = yaml.load(data, Loader=yaml.FullLoader)
    return data


def get_norm(norm_type):
    if norm_type == "BatchNorm":
        norm = nn.BatchNorm2d
    elif norm_type == "IdentityNorm":
        norm = resnet.IdentityNorm
    else:
        raise NotImplementedError

    return norm


def get_activation(act_type):
    if act_type == "Identity":
        act = resnet.Identity()
    elif act_type == "Relu":
        act = nn.ReLU(inplace=True)
    else:
        raise NotImplementedError

    return act


def get_transformations(argumentation_configs):
    trans = []
    trans_available = {
        "zero_padding": transforms.Pad,
        "random_crop": transforms.RandomCrop,
    }
    for key in argumentation_configs:
        if key in trans_available.keys():
            trans.append(trans_available[key](argumentation_configs[key]))
    return trans


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
