import os
import importlib
import torch.nn as nn
from functools import partial

from classification import utils


MODEL_MODULES = {
    "classification/models": "classification.models.",
    "gan/models"           : "gan.models.",
    "nn_module/conv"       : "nn_module.conv.",
    "nn_module/norm"       : "nn_module.norm.",
    "nn_module/act"        : "nn_module.act.",
}


class Register:

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            raise Exception(f"Key {key} already in registry {self._name}.")
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        return self._dict.keys()


def import_all_modules_for_register():
    """Import all modules for register."""
    for base_dir in MODEL_MODULES.keys():
        files = os.listdir(os.path.join(os.getcwd(), base_dir))
        for name in files:
            name = name.split(".")[0]
            full_name = MODEL_MODULES[base_dir] + name
            importlib.import_module(full_name)


def make_network(model_config):
    """
    Construct the model.
    The Register can automatically load corresponding model
    using the model name once it was registered in the class definition.
    Each model class (under "classification/models") defines its own "make_network" method to parse the args.
    So you can see the model's "make_network" method to find out the valid args for the model.
    """
    for name in name_to_model.keys():
        if name in model_config:
            sub_configs = model_config[name]
            model = name_to_model[name].make_network(sub_configs)
            return model, sub_configs


def get_norm(norm_type):
    """
    Helper function to get norm Class using their name with args.

    Parameters
    ----------
    norm_type: str or dict
      If norm_type is str, it is inferred as the name of norm method.
      We get the Class using the dict NAME_TO_NORMS.
      Example:
        norm: "BatchNorm"

      If norm_type is dict, it is inferred as norm name with some
      initialization args.
      We can use this method to provide some args which are irrelevant to
      input size, such as eps and momentum for BatchNorm.
      Examples:
        norm:
          BatchNorm:
            eps: 1e-5,
            momentum: 0.1
    """
    def get_norm_by_name(name):
        if name in NAME_TO_NORMS:
            return NAME_TO_NORMS[name]
        else:
            raise NotImplementedError(f"The specified {name} is not implemented. The available values are {NAME_TO_NORMS.keys()}")

    if norm_type is None:
        return get_norm_by_name("Identity")
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


def get_activation(act_type):
    def get_act_by_name(name):
        if name in NAME_TO_ACTS:
            return NAME_TO_ACTS[name]
        else:
            raise NotImplementedError(f"The specified {name} is not implemented. The available values are {NAME_TO_ACTS.keys()}")

    if act_type is None:
        return get_act_by_name("Identity")
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


def get_conv(configs, conv_name=None):
    conv_name = "conv" if conv_name is None else conv_name
    conv_type = configs[conv_name] if conv_name in configs else "Conv2d"
    if isinstance(conv_type, str):
        assert conv_type in NAME_TO_CONVS, f"Conv type {conv_type} is not supported!"
        conv = NAME_TO_CONVS[conv_type]
        conv = conv.get_conv(configs={})  # As conv_type is str, we don't need to pass any args.
    else:
        conv_name = list(conv_type.keys())
        if len(conv_name) > 1:
            raise Exception("Not support more than one Conv block!")
        if len(conv_name) == 0:
            raise Exception("At least one Conv block must be specified!")
        conv_name = conv_name[0]
        if conv_name not in NAME_TO_CONVS.keys():
            raise NotImplementedError(f"The specified {conv_name} is not implemented. The available values are {NAME_TO_CONVS.keys()}")
        conv_configs = conv_type[conv_name]
        conv = NAME_TO_CONVS[conv_name]
        conv = conv.get_conv(conv_configs)

    return conv


# Registration for model
name_to_model = Register("name_to_model")

# Registration for norm
NAME_TO_NORMS = Register("name_to_norms")
NAME_TO_NORMS["BatchNorm"] = nn.BatchNorm2d

# Registration for activation
NAME_TO_ACTS = Register("name_to_acts")
NAME_TO_ACTS["IdentityAct"] = utils.Identity
NAME_TO_ACTS["Relu"] = nn.ReLU
NAME_TO_ACTS["ReLU"] = nn.ReLU
NAME_TO_ACTS["LeakyRelu"] = nn.LeakyReLU
NAME_TO_ACTS["LeakyReLU"] = nn.LeakyReLU

# Registration for conv
NAME_TO_CONVS = Register("name_to_convs")

import_all_modules_for_register()