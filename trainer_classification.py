import os
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader
from ptflops import get_model_complexity_info
from torchvision import datasets

import register
from classification import utils
from classification.dataset import DatasetCls

if __name__ == "__main__":
    # Kindly print the current path of your env.
    # So you can quickly find the config file path error when it occurs.
    print(f"The current path is: {os.getcwd()}")

    # Load configs
    parser = argparse.ArgumentParser(description="Trainer for classification task.")
    parser.add_argument('--config_file', type=str,
                        default="classification/configs/MobileOne/MobileOne_20_CIFAR10_EXP.yaml",
                        help="Path of config file.")
    config_file_path = parser.parse_args().config_file
    configs = utils.load_yaml_file(config_file_path)

    # Create output dict if it does not exist
    output_path = os.path.join(configs["Train"]["output"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # This is logger. All training info will be stored in it.
    log = utils.Logger(os.path.join(output_path, "train.log"))

    # Construct argumentation methods.
    # You can use any argumentation methods supported by PyTorch
    # simply setting the tag "Argumentation" in the config file.
    # See config files in the "classification/configs" dict for example.
    train_trans = []
    if "Argumentation" in configs:
        train_trans = utils.get_transformations(configs["Argumentation"])
        if "mean" in configs["Argumentation"] and "std" in configs["Argumentation"]:
            mean, std = configs["Argumentation"]["mean"], configs["Argumentation"]["std"]
        else:
            # If mean OR std is not specified, we use the default values from ImageNet
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    trans = [
        # You need to specify the image size by setting "h" and "w" in the config file
        transforms.Resize((configs["Dataset"]["h"], configs["Dataset"]["w"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    trn_trans = transforms.Compose(train_trans + trans)
    tst_trans = transforms.Compose(trans)

    # Set your dataset. There are two ways to do this.
    # First, you can use the dataset name in the config file (now only CIFAR10 is included).
    # Second, you can set the training data path and test data path explicitly.
    if "name" in configs["Dataset"]:
        if configs["Dataset"]["name"] == "CIFAR10":
            trn_data = datasets.CIFAR10(root=configs["Dataset"]["root_path"], train=True, transform=trn_trans, download=True)
            tst_data = datasets.CIFAR10(root=configs["Dataset"]["root_path"], train=False, transform=tst_trans, download=True)
        else:
            raise NotImplementedError
    else:
        trn_data = DatasetCls(configs["Dataset"]["trn_path"], transforms=trn_trans)
        tst_data = DatasetCls(configs["Dataset"]["tst_path"], transforms=tst_trans)

    # Construct the dataloader
    num_workers = configs["Dataset"]["num_workers"] if "num_workers" in configs["Dataset"] else 1
    pin_memory = configs["Dataset"]["pin_memory"] if "pin_memory" in configs["Dataset"] else False
    batch_size = configs["Dataset"]["batch_size"]
    trn_loader = DataLoader(trn_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=True,
                            pin_memory=pin_memory)
    tst_loader = DataLoader(tst_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False,
                            pin_memory=pin_memory)

    # Construct the model.
    # The Register can automatically load corresponding model
    # using the model name once it was registered in the class definition.
    # Each model class (under "classification/models") defines its own "make_network" method to parse the args.
    # So you can see the model's "make_network" method to find out the valid args for the model.
    model, model_configs = register.make_network(configs["Model"])

    # Infer the device used for training.
    # The simplest way to set the value is:
    # 1) set to "cpu" if cpu is used;
    # 2) set to a list of IDs (int) if GPUs are used.
    # E.g. you can set to [0] is only one GPU is used.
    device = configs["Train"]["device"]
    device_id = None
    if isinstance(device, str):
        assert device in ["cuda", "cpu"]
        if device == "cuda":
            device_id = "cuda:0"
            device_ids = [0]
    elif isinstance(device, list):
        device_id = f"cuda:{device[0]}"
        device_ids = list(device)
        device = "cuda"

    # Load a pre-trained model if "snapshot" is specified
    if "snapshot" in configs["Train"]:
        model.load_state_dict(torch.load(configs["Train"]["snapshot"])["model"])

    # Set multi-GPU mode if more than one GPU used.
    if device == "cuda":
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device_id)

    # Set the model initialization method.
    # You can use any initialization methods supported by PyTorch
    # simply setting the tag "Init" in the config file.
    # See config files in the "classification/configs" dict for example.
    if "Init" in configs["Model"]:
        utils.init_nn(model, configs["Model"]["Init"])

    # If specified, save the L2 norm of gradients at each layer
    # into the output model file for further analysis.
    keep_gradients = False
    if "keep_gradients" in configs["Train"] and configs["Train"]["keep_gradients"] is True:
        gradients_dic = defaultdict(list)
        keep_gradients = True

    # Calculate the model compexity.
    # The current method is based on the ptflops.
    # But I found that it cannot get correct results when multi-GPUs are used.
    input_shape = (model_configs["in_channels"],
                   configs["Dataset"]["h"],
                   configs["Dataset"]["w"])
    macs, params = get_model_complexity_info(model,
                                             input_shape,
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    log.logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    log.logger.info('{:<30}  {:<8}'.format('Number of parameters    : ', params))

    # Set loss function
    num_classes = model_configs["num_classes"]
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device_id) if device == "cuda" else criterion

    # Set optimizer.
    # You can use any optimizer supported by PyTorch
    # simply setting the tag "OPT" in the config file.
    # See config files in the "classification/configs" dict for example.
    optimizer = utils.get_optimizer(model.parameters(), configs["Model"]["OPT"])

    # Set scheduler.
    # You can use any scheduler supported by PyTorch
    # simply setting the tag "Scheduler" in the config file.
    # See config files in the "classification/configs" dict for example.
    scheduler = None
    if "Scheduler" in configs["Model"]:
        scheduler = utils.get_scheduler(optimizer, configs["Model"]["Scheduler"])

    # Following training scripts are verbose.
    # Of ofcourse we can make it clear to warp some codes.
    # But as most codes are used only here, warping makes no sense.
    trn_error_list, tst_error_list, trn_loss_list, tst_loss_list = [], [], [], []
    iterations, best_error, best_iter, trn_loss, trn_pos = 0, 1., 0, 0., 0.
    save_freq = configs["Train"]["save_freq"]

    while True:
        for x, y in trn_loader:
            if iterations == configs["Train"]["iterations"]:
                break
            model.train()
            optimizer.zero_grad()
            if device == "cuda":
                x = x.to(device_id)
                y = y.to(device_id)

            pred = model(x)
            if num_classes == 2:
                pred = pred.squeeze(-1)
                y = y.float()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            if num_classes == 2:
                trn_pos += (pred.gt(0.5) == y).sum().cpu()
            else:
                trn_pos += (pred.argmax(dim=-1) == y).sum().cpu()
            trn_loss += loss.item()
            if scheduler is not None:
                scheduler.step()

            iterations += 1
            if iterations % save_freq == 0:
                if keep_gradients:
                    for name, layer in model.named_modules():
                        if isinstance(layer, nn.Conv2d):
                            length = 1
                            for s in layer.weight.grad.size():
                                length *= s
                            g = torch.linalg.vector_norm(layer.weight.grad) / length
                            gradients_dic[name].append(g.item())

                trn_loss = trn_loss / save_freq
                trn_loss_list.append(trn_loss)
                trn_error = 1 - trn_pos / (batch_size * save_freq)
                trn_error_list.append(trn_error)
                log.logger.info(f"The training loss at {iterations}-th iteration : {trn_loss}")
                log.logger.info(f"The training error at {iterations}-th iteration: {trn_error}")
                trn_loss = 0.
                trn_pos = 0.

                tst_loss = 0.
                tst_pos = 0.
                model.eval()
                with torch.no_grad():
                    for x, y in tst_loader:                 
                        if device == "cuda":
                            x = x.to(device_id)
                            y = y.to(device_id)

                        pred = model(x)
                        if num_classes == 2:
                            pred = pred.squeeze(-1)
                            y = y.float()
                        loss = criterion(pred, y)
                        tst_loss += loss.item() * x.size(0)
                        if num_classes == 2:
                            trn_pos += (pred.gt(0.5) == y).sum().cpu()
                        else:
                            trn_pos += (pred.argmax(dim=-1) == y).sum().cpu()

                    tst_error = 1 - tst_pos / len(tst_data)
                    tst_loss = tst_loss / len(tst_data)
                    tst_loss_list.append(tst_loss)
                    tst_error_list.append(tst_error)
                    log.logger.info(f"The test loss at {iterations}-th iteration : {tst_loss}")
                    log.logger.info(f"The test error at {iterations}-th iteration: {tst_error}")

                    # save best
                    if tst_error < best_error:
                        best_error = tst_error
                        best_iter = iterations
                        if device == "cuda":
                            state_dic = model.module.state_dict()
                        else:
                            state_dic = model.state_dict()
                        state = {
                            "model": state_dic,
                            "opt": optimizer.state_dict(),
                            "iterations": iterations,
                            "trn_loss": trn_loss_list,
                            "tst_loss": tst_loss_list,
                            "trn_error": trn_error_list,
                            "tst_error": tst_error_list,
                        }
                        torch.save(state, os.path.join(output_path, "best.pth"))

                    # save last
                    if device == "cuda":
                        state_dic = model.module.state_dict()
                    else:
                        state_dic = model.state_dict()
                    state = {
                        "model": state_dic,
                        "opt": optimizer.state_dict(),
                        "iterations": iterations,
                        "trn_loss": trn_loss_list,
                        "tst_loss": tst_loss_list,
                        "trn_error": trn_error_list,
                        "tst_error": tst_error_list
                    }
                    if keep_gradients:
                        state["gradients"] = gradients_dic
                    torch.save(state, os.path.join(output_path, "last.pth"))

                    log.logger.info(f"The best iteration at {iterations}-th iteration: {best_iter}")
                    log.logger.info(f"The best error at {iterations}-th iteration    : {best_error}")
                    log.logger.info("")

                    plt.figure(figsize=(20, 8), dpi=80)
                    epoch_list = [i + 1 for i in range(len(trn_loss_list))]
                    plt.plot(epoch_list, trn_error_list, color="red", label="training_error")
                    plt.plot(epoch_list, tst_error_list, color="blue", label="test_error")
                    plt.xlabel(f"iterations x{save_freq}")
                    plt.ylabel("error")
                    plt.legend(loc="upper right")
                    plt.savefig(os.path.join(configs["Train"]["output"], "train_test_curve.jpg"))
                    plt.close()

        if iterations == configs["Train"]["iterations"]:
            break

    if "deploy" in configs["Train"] and configs["Train"]["deploy"]:
        for name, layer in model.named_modules():
            method_list = [func for func in dir(layer) if callable(getattr(layer, func))]
            if "switch_to_deploy" in method_list:
                layer.switch_to_deploy()

        method_list = [func for func in dir(model) if callable(getattr(model, func))]
        if "switch_to_deploy" in method_list:
            model.switch_to_deploy()

        input_shape = (model_configs["in_channels"],
                       configs["Dataset"]["h"],
                       configs["Dataset"]["w"])
        macs_deploy, params_deploy = get_model_complexity_info(model,
                                                               input_shape,
                                                               as_strings=True,
                                                               print_per_layer_stat=True,
                                                               verbose=True)
        log.logger.info('{:<30}{:<8}'.format('Computational complexity (original): ', macs))
        log.logger.info('{:<30}{:<8}'.format('Number of parameters     (original): ', params))
        log.logger.info('{:<30}{:<8}'.format('Computational complexity (deploy)  : ', macs_deploy))
        log.logger.info('{:<30}{:<8}'.format('Number of parameters     (deploy)  : ', params_deploy))

        if "deploy_test" in configs["Train"] and configs["Train"]["deploy_test"]:
            tst_loss = 0.
            tst_pos = 0.
            model.eval()
            with torch.no_grad():
                for x, y in tst_loader:
                    if device == "cuda":
                        x = x.to(device_id)
                        y = y.to(device_id)

                    pred = model(x)
                    if num_classes == 2:
                        pred = pred.squeeze(-1)
                        y = y.float()
                    loss = criterion(pred, y)
                    tst_loss += loss.item() * x.size(0)
                    if num_classes == 2:
                        trn_pos += (pred.gt(0.5) == y).sum().cpu()
                    else:
                        trn_pos += (pred.argmax(dim=-1) == y).sum().cpu()

                tst_error = 1 - tst_pos / len(tst_data)
                tst_loss = tst_loss / len(tst_data)
                tst_loss_list.append(tst_loss)
                tst_error_list.append(tst_error)
                log.logger.info(f"The test loss at{iterations}-th iteration :{tst_loss}")
                log.logger.info(f"The test error at{iterations}-th iteration:{tst_error}")

        # save last
        if device == "cuda":
            state_dic = model.module.state_dict()
        else:
            state_dic = model.state_dict()
        state = {
            "model": state_dic
        }
        torch.save(state, os.path.join(output_path, "model_deploy_last.pth"))
