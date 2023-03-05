import os
import cv2
import copy
import math
import torch
import argparse
import torch.nn.functional as F
from collections import defaultdict
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

import register
from classification import utils
from gan.dataset import GANDataset
from gan.evaluation import calculate_frechet_distance


def back(img):
    img_tran = copy.deepcopy(img)
    img_tran[2] = (img[0] * 0.5 + 0.5) * 255
    img_tran[1] = (img[1] * 0.5 + 0.5) * 255
    img_tran[0] = (img[2] * 0.5 + 0.5) * 255
    img_tran = torch.clamp(img_tran, min=0, max=255)
    return img_tran.permute(1, 2, 0)


def cal_trn_imgs_FID_statistics():
    fid_mean, fid_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    fid_trans = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=fid_mean, std=fid_std)
    ])
    fid_data = GANDataset(configs["Dataset"]["trn_path"], configs["Dataset"]["vec_dim"], transforms=fid_trans)
    fid_loader = DataLoader(fid_data,
                            batch_size=metric_configs["InceptionScore_FID"]["batch_size"],
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False,
                            pin_memory=pin_memory)
    device_id, _, device = utils.parse_device(model_configs["Train"]["device"])
    feats = []
    with torch.no_grad():
        for data in fid_loader:
            img = data["real"].to(device_id)
            _, _, feat = inception3(img)
            feats.append(feat)
    feats = torch.cat(feats, dim=0)
    mu = torch.mean(feats, dim=0)
    sigma = torch.cov(feats.t())
    return mu, sigma


def cal_metrics(mu_w, sigma_w):
    metrics = {}
    if "Metric" in model_configs["Train"]:
        metric_configs = model_configs["Train"]["Metric"]
        data_temp = GANDataset(configs["Dataset"]["trn_path"], configs["Dataset"]["vec_dim"], transforms=trn_trans)
        data_temp = DataLoader(data_temp,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers,
                               drop_last=True,
                               pin_memory=pin_memory)

        if "InceptionScore_FID" in metric_configs:
            n_images = metric_configs["InceptionScore_FID"]["n_images"]
            n_repeat = metric_configs["InceptionScore_FID"]["n_repeat"]
            kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
            resize_op = transforms.Resize(size=299)
            scores, feats = [], []

            with torch.no_grad():
                for _ in range(n_repeat):
                    count = 0
                    preds = []
                    while count < n_images:
                        for data in data_temp:
                            model.set_input(data)
                            model.forward()
                            imgs_fake = model.imgs_fake
                            count += batch_size
                            x_ch0 = torch.unsqueeze(imgs_fake[:, 0], 1) * (0.5 / 0.229) + (0.5 - 0.485) / 0.229
                            x_ch1 = torch.unsqueeze(imgs_fake[:, 1], 1) * (0.5 / 0.224) + (0.5 - 0.456) / 0.224
                            x_ch2 = torch.unsqueeze(imgs_fake[:, 2], 1) * (0.5 / 0.225) + (0.5 - 0.406) / 0.225
                            imgs_fake = torch.cat((x_ch0, x_ch1, x_ch2), 1)
                            imgs_fake = resize_op(imgs_fake)

                            inception_batch = metric_configs["InceptionScore_FID"]["batch_size"]
                            for i in range(1, int(batch_size / inception_batch + 1)):
                                input_imgs = imgs_fake[int((i - 1) * inception_batch): int(i * inception_batch)]
                                pred, _, feat = inception3(input_imgs)
                                preds.append(pred.cpu().detach())
                                feats.append(feat.cpu().detach())
                            if count >= n_images:
                                break

                    preds = torch.cat(preds, dim=0)
                    preds = F.softmax(preds, dim=-1)
                    p_y = torch.mean(preds, dim=0)

                    preds = torch.log(preds)
                    p_y = torch.log(p_y).unsqueeze(0)
                    p_y = p_y.repeat(preds.size(0), 1)
                    score = math.exp(kl_div(preds, p_y))
                    scores.append(score)

                metrics["IS"] = sum(scores) / len(scores)

                # Compute FID score
                feats = torch.cat(feats, dim=0)[:int(n_images * n_repeat)]
                mu = torch.mean(feats, dim=0).numpy()
                sigma = torch.cov(feats.t()).numpy()
                mu_w = mu_w.cpu().detach().numpy()
                sigma_w = sigma_w.cpu().detach().numpy()
                metrics["FID"] = calculate_frechet_distance(mu, sigma, mu_w, sigma_w)
    return metrics


if __name__ == "__main__":
    # Kindly print the current path of your env.
    # So you can quickly find the config file path error when it occurs.
    print(f"The current path is: {os.getcwd()}")

    # Load configs
    parser = argparse.ArgumentParser(description="Trainer for GAN task.")
    parser.add_argument('--config_file', type=str,
                        default="gan/configs/CIFAR10/DCGAN_CIFAR10_small.yaml",
                        help="Path of config file.")
    config_file_path = parser.parse_args().config_file
    configs = utils.load_yaml_file(config_file_path)

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
            # If mean OR std is not specified, we use the default values to map the
            # pixels to range [-1, 1]
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    trans = [
        # You need to specify the image size by setting "h" and "w" in the config file
        transforms.Resize((configs["Dataset"]["h"], configs["Dataset"]["w"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    trn_trans = transforms.Compose(train_trans + trans)
    tst_trans = transforms.Compose(trans)

    # Set your dataset
    trn_data = GANDataset(configs["Dataset"]["trn_path"], configs["Dataset"]["vec_dim"], transforms=trn_trans)
    tst_data = GANDataset(configs["Dataset"]["tst_path"], configs["Dataset"]["vec_dim"], transforms=tst_trans)

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
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=True,
                            pin_memory=pin_memory)

    # Construct the model.
    # The Register can automatically load corresponding model
    # using the model name once it was registered in the class definition.
    # Each model class (under "classification/models") defines its own "make_network" method to parse the args.
    # So you can see the model's "make_network" method to find out the valid args for the model.
    model, model_configs = register.make_network(configs["Model"])

    # Create output dict if it does not exist
    output_path = os.path.join(model_configs["Train"]["output"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # This is logger. All training info will be stored in it.
    log = utils.Logger(os.path.join(output_path, "train.log"))

    # Print model info
    model.setup(configs, log)
    model.train()

    # Load Inception3 model if it is needed to compute evaluation metrics
    best_metrics = {}
    history_metrics = defaultdict(list)
    inception3 = None
    if "Metric" in model_configs["Train"]:
        metric_configs = model_configs["Train"]["Metric"]
        if "InceptionScore_FID" in metric_configs:
            best_metrics["IS"] = 1
            best_metrics["FID"] = float("inf")
            from classification.models.inception3 import Inception3
            inception3 = Inception3(mode="test")
            inception3.load_state_dict(torch.load(metric_configs["InceptionScore_FID"]["inception3_path"]))
            inception3 = utils.set_device(inception3, *utils.parse_device(model_configs["Train"]["device"]))
            inception3.eval()
            mu_w, sigma_w = cal_trn_imgs_FID_statistics()

    iterations = 1
    while iterations < model_configs["Train"]["iterations"]:
        for data in trn_loader:
            if iterations == model_configs["Train"]["iterations"]:
                break

            model.set_input(data)
            model.optimize_parameters()

            if iterations % model_configs["Train"]["print_freq"] == 0:
                losses = model.get_current_losses()
                for loss in losses.keys():
                    log.logger.info(f"The {loss} at {iterations}-th iteration: {losses[loss]}")
                log.logger.info("\n")

            if iterations % model_configs["Train"]["save_freq"] == 0:
                model.eval()
                save_to = os.path.join(output_path, str(iterations))
                if not os.path.exists(save_to):
                    os.makedirs(save_to)
                model.save_networks(save_to)
                model.save_figures(save_to)
                imgs = model.compute_info(tst_loader)

                for name, img in imgs:
                    if not os.path.exists(os.path.join(output_path, str(iterations), "imgs")):
                        os.makedirs(os.path.join(output_path, str(iterations), "imgs"))
                    img_path = os.path.join(output_path, str(iterations), "imgs", f"{name}.jpg")
                    img = back(img.squeeze(0)).cpu().numpy()
                    cv2.imwrite(img_path, img)

                if "Metric" in model_configs["Train"]:
                    metric_configs = model_configs["Train"]["Metric"]
                    if "InceptionScore_FID" in metric_configs:
                        metrics = cal_metrics(mu_w, sigma_w)
                        for k, v in metrics.items():
                            if k in ["IS"] and v > best_metrics[k]:
                                best_metrics[k] = v
                            elif k in ["FID"] and v < best_metrics[k]:
                                best_metrics[k] = v
                            history_metrics[k].append(v)
                            log.logger.info(f"The metric {k} at {iterations}-th is     : {v}")
                            log.logger.info(f"The best metric {k} at {iterations}-th is: {best_metrics[k]}")
                            iteration_list = [i for i in range(len(history_metrics[k]))]
                            save_freq = model_configs["Train"]["save_freq"]
                            utils.draw_line_figure(data_list=[[iteration_list, history_metrics[k], "red", k]],
                                                   figsize=(20, 8),
                                                   dpi=80,
                                                   x_label=f"iterations x{save_freq}",
                                                   y_label=k,
                                                   legend_loc="upper right",
                                                   save_path=os.path.join(model_configs["Train"]["output"], f"{k}.jpg"))
                model.train()

            iterations += 1
