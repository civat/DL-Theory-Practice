import os
import importlib

MODEL_MODULES = ["classification", "gan"]


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
    for base_dir in MODEL_MODULES:
        files = os.listdir(os.path.join(os.getcwd(), base_dir, "models"))
        for name in files:
            name = name.split(".")[0]
            full_name = base_dir + ".models." + name
            importlib.import_module(full_name)


name_to_model = Register("name_to_model")
import_all_modules_for_register()


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
