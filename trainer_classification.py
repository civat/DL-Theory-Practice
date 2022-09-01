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
    print(f"The current path is: {os.getcwd()}")
    parser = argparse.ArgumentParser(description="Trainer for classification task.")
    parser.add_argument('--config_file', type=str,
                        default="classification/configs/ResNet/ResNet_20-Layers_CIFAR10_EXP.yaml",
                        help="Path of config file.")

    config_file_path = parser.parse_args().config_file
    configs = utils.load_yaml_file(config_file_path)
    batch_size = configs["Dataset"]["batch_size"]

    output_path = os.path.join(configs["Train"]["output"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log = utils.Logger(os.path.join(output_path, "train.log"))

    # transformations
    train_trans = []
    if "Argumentation" in configs:
        train_trans = utils.get_transformations(configs["Argumentation"])
        if "mean" in configs["Argumentation"] and "std" in configs["Argumentation"]:
            mean, std = configs["Argumentation"]["mean"], configs["Argumentation"]["std"]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    trans = [
        transforms.Resize((configs["Dataset"]["h"], configs["Dataset"]["w"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    trn_trans = transforms.Compose(train_trans + trans)
    tst_trans = transforms.Compose(trans)

    if "name" in configs["Dataset"]:
        if configs["Dataset"]["name"] == "CIFAR10":
            trn_data = datasets.CIFAR10(root=configs["Dataset"]["root_path"], train=True, transform=trn_trans, download=True)
            tst_data = datasets.CIFAR10(root=configs["Dataset"]["root_path"], train=False, transform=tst_trans, download=True)
        else:
            raise NotImplementedError
    else:
        trn_data = DatasetCls(configs["Dataset"]["trn_path"], transforms=trn_trans)
        tst_data = DatasetCls(configs["Dataset"]["tst_path"], transforms=tst_trans)

    num_workers = configs["Dataset"]["num_workers"] if "num_workers" in configs["Dataset"] else 1
    pin_memory = configs["Dataset"]["pin_memory"] if "pin_memory" in configs["Dataset"] else False
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

    for name in register.name_to_model.keys():
        if name in configs["Model"]:
            model_name = name
            sub_configs = configs["Model"][model_name]
            model = register.name_to_model[model_name].make_network(sub_configs)

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

    if "snapshot" in configs["Train"]:
        model.load_state_dict(torch.load(configs["Train"]["snapshot"])["model"])

    if device == "cuda":
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device_id)

    if "Init" in configs["Model"]:
        utils.init_nn(model, configs["Model"]["Init"])

    keep_gradients = False
    if "keep_gradients" in configs["Train"] and configs["Train"]["keep_gradients"] is True:
        gradients_dic = defaultdict(list)
        keep_gradients = True

    input_shape = (sub_configs["in_channels"],
                   configs["Dataset"]["h"],
                   configs["Dataset"]["w"])
    macs, params = get_model_complexity_info(model,
                                             input_shape,
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    log.logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    log.logger.info('{:<30}  {:<8}'.format('Number of parameters    : ', params))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device_id) if device == "cuda" else criterion

    # Set optimizer
    optimizer = utils.get_optimizer(model.parameters(), configs["Model"]["OPT"])

    # Set scheduler
    scheduler = None
    if "Scheduler" in configs["Model"]:
        scheduler = utils.get_scheduler(optimizer, configs["Model"]["Scheduler"])

    iterations = 0
    trn_error_list, tst_error_list, trn_loss_list, tst_loss_list = [], [], [], []
    best_error = 1
    best_iter = 0
    trn_loss = 0.
    trn_pos = 0.
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
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
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
                        loss = criterion(pred, y)
                        tst_loss += loss.item() * x.size(0)
                        tst_pos += (pred.argmax(dim=-1) == y).sum().cpu()

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

        input_shape = (sub_configs["in_channels"],
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
                    loss = criterion(pred, y)
                    tst_loss += loss.item() * x.size(0)
                    tst_pos += (pred.argmax(dim=-1) == y).sum().cpu()

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
