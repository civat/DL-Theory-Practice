import os
import cv2
import copy
import torch
import argparse
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

import register
from classification import utils
from gan.datasets import GANDataset


def back(img):
    img_tran = copy.deepcopy(img)
    img_tran[2] = (img[0] * 0.5 + 0.5) * 255
    img_tran[1] = (img[1] * 0.5 + 0.5) * 255
    img_tran[0] = (img[2] * 0.5 + 0.5) * 255
    img_tran = torch.clamp(img_tran, min=0, max=255)
    return img_tran.permute(1, 2, 0)


if __name__ == "__main__":
    # Kindly print the current path of your env.
    # So you can quickly find the config file path error when it occurs.
    print(f"The current path is: {os.getcwd()}")

    # Load configs
    parser = argparse.ArgumentParser(description="Trainer for GAN task.")
    parser.add_argument('--config_file', type=str,
                        default="classification/configs/MobileOne/MobileOne_20_CIFAR10_EXP.yaml",
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

    # Create output dict if it does not exist
    output_path = os.path.join(configs["Train"]["output"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # This is logger. All training info will be stored in it.
    log = utils.Logger(os.path.join(output_path, "train.log"))

    # Print model info
    model.setup(configs, log)
    model.train()

    iterations = 1
    while True:
        for data in trn_loader:
            if iterations == configs["Train"]["iterations"]:
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
                    img = back(img.squeeze(0).cpu().numpy())
                    cv2.imwrite(img_path, img)
                model.train()

            iterations += 1
