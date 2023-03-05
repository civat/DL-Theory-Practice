import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info

import register
from gan.gan_loss import GANLoss
from gan.models.base_model import BaseModel
from gan.gan_loss import cal_gradient_penalty
from classification import utils


@register.name_to_model.register("GAN")
class GAN(BaseModel):

    def __init__(self, configs):
        super(GAN, self).__init__(configs)

        # Specify the training losses you want to print out.
        # "G_A" means "Adversary loss of Generator".
        # "D_A" means "Adversary loss of Discriminator".
        self.loss_names = ["G_A", "D_A"]

        # Specify the models you want to save to the disk.
        # "G" means "Generator".
        # "D" means "Discriminator".
        self.model_names = ["G", "D"]

        # Define generator and discriminator
        self.G, self.G_configs = register.make_network(configs["G"])
        self.D, self.D_configs = register.make_network(configs["D"])

        # Infer the device used for training
        self.device_id, self.device_ids, self.device = utils.parse_device(configs["Train"]["device"])

        # Define the loss function
        self.gan_loss_name = GANLoss(configs["Train"]["GAN_loss"])
        if isinstance(self.gan_loss_name, dict):
            self.gan_loss_name = list(self.gan_loss_name.keys())
            if len(self.gan_loss_name) != 1:
                raise Exception("Only one loss can (must) be specified!")
            self.gan_loss_name = self.gan_loss_name[0]
            self.gan_loss_args = configs["Train"]["GAN_loss"][self.gan_loss_name]

        self.gan_loss = GANLoss(self.gan_loss_name)
        self.gan_loss.to(self.device_id)

        # Define optimizers
        self.optimizer_G = utils.get_optimizer(self.G.parameters(), configs["G"]["OPT"])
        self.optimizer_D = utils.get_optimizer(self.D.parameters(), configs["D"]["OPT"])

        self.scheduler_G = None
        if "Scheduler" in configs["G"]:
            self.scheduler_G = utils.get_scheduler(self.optimizer_G, configs["G"]["Scheduler"])
        self.scheduler_D = None
        if "Scheduler" in configs["D"]:
            self.scheduler_D = utils.get_scheduler(self.optimizer_D, configs["D"]["Scheduler"])

        if "Init" in configs["G"]:
            utils.init_nn(self.G, configs["G"]["Init"])
        if "Init" in configs["D"]:
            utils.init_nn(self.D, configs["D"]["Init"])

        self.n_save_imgs = configs["Train"]["save_imgs"]  # number of generated images to save
        self.n_critics = configs["Train"]["n_critics"] if "n_critics" in configs["Train"] else 1

        # used for drawing figures
        self.iterations = 1
        self.iter_list = []
        self.loss_D_list = []
        self.loss_G_list = []
        self.save_freq = configs["Train"]["save_freq"]
   
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters
        ----------
        input: dict 
          Include the data itself and its metadata information.
        """
        self.imgs_real = input["real"].to(self.device_id)
        self.vecs_rand = input["vec"].to(self.device_id)
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.imgs_fake = self.G(self.vecs_rand)
    
    def backward_D(self):
        """
        Calculate GAN loss for the discriminator.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = self.D(self.imgs_real)
        loss_D_real = self.gan_loss(pred_real, True)

        # Fake
        pred_fake = self.D(self.imgs_fake.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)

        # Combined loss and calculate gradients
        self.loss_D_A = loss_D_real + loss_D_fake
        if self.gan_loss_name == "wgangp":
            gradient_penalty = cal_gradient_penalty(self.D, self.imgs_real, self.imgs_fake.detach(), self.device_id, **self.gan_loss_args)
            self.loss_D_A += gradient_penalty
        self.loss_D_A.backward()

    def backward_G(self):
        """
        Calculate GAN loss for the generator.
        We also call loss_G.backward() to calculate the gradients.
        """
        self.loss_G_A = self.gan_loss(self.D(self.imgs_fake), True)
        self.loss_G_A.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""  
        # forward
        self.forward()

        # D
        self.set_requires_grad([self.D], True)
        self.optimizer_D.zero_grad()             # set D's gradients to zero
        self.backward_D()                        # calculate gradients for D
        self.optimizer_D.step()                  # update D's weights

        # G
        if self.iterations % self.n_critics == 0:
            self.set_requires_grad([self.D], False)  # D require no gradients when optimizing G
            self.optimizer_G.zero_grad()             # set G's gradients to zero
            self.backward_G()                        # calculate gradients for G
            self.optimizer_G.step()                  # update G's weights

        if self.scheduler_G is not None:
            self.scheduler_G.step()
        if self.scheduler_D is not None:
            self.scheduler_D.step()

        # keep info for drawing figures
        if self.iterations % self.save_freq == 0:
            self.iter_list.append(self.iterations)
            self.loss_D_list.append(self.loss_D_A.item())
            self.loss_G_list.append(self.loss_G_A.item())
        self.iterations += 1

    def compute_info(self, data_loader):
        count = 0
        imgs = []
        with torch.no_grad():
            for data in data_loader:
                vecs_rand = data["vec"].to(self.device_id)
                fake = self.G(vecs_rand)
                imgs.append((f"{count}_fak", fake))
                count += 1
                if count % self.n_save_imgs == 0:
                    break
        return imgs

    def setup(self, configs, log):
        # Calculate the model compexity.
        # The current method is based on the ptflops.
        # ptflops cannot get correct results when multi-gpus are used.
        # So we first move models to only one GPU.
        self.G = self.G.to(self.device_id)
        self.D = self.D.to(self.device_id)

        log.logger.info("Model complexity of Generator:")
        input_shape = (configs["Dataset"]["vec_dim"],)
        utils.cal_model_complexity(self.G, input_shape, log)

        log.logger.info("Model complexity of Discriminator:")
        input_shape = (configs["Dataset"]["c"],
                       configs["Dataset"]["h"],
                       configs["Dataset"]["w"])
        utils.cal_model_complexity(self.D, input_shape, log)

        self.G = utils.set_device(self.G, self.device_id, self.device_ids, self.device)
        self.D = utils.set_device(self.D, self.device_id, self.device_ids, self.device)

    def save_figures(self, output_path):
        plt.figure(figsize=(20, 8), dpi=80)
        plt.plot(self.iter_list, self.loss_D_list, color="red", label="D loss")
        plt.plot(self.iter_list, self.loss_G_list, color="blue", label="G loss")
        plt.xlabel(f"iterations")
        plt.ylabel("loss")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(output_path, "train.jpg"))
        plt.close()

    @staticmethod
    def make_network(configs):
        return GAN(configs)