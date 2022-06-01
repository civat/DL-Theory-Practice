import yaml
import torch.nn as nn
import torchvision.transforms as transforms

from Classification import resnet


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
        "padding": transforms.Pad,
        "random_crop": transforms.RandomCrop,
    }
    for key in argumentation_configs:
        if key in trans_available.keys():
            trans.append(trans_available[key](argumentation_configs[key]))
    return trans
