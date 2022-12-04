import torch

from base_model import BaseModel


class GAN(BaseModel):

    def __init__(self, configs):
        super(GAN, self).__init__(configs)

        # specify the training losses you want to print out.
        # "G_A" means "Adversary loss of Generator".
        # "D_A" means "Adversary loss of Discriminator".
        self.loss_names = ["G_A", "D_A"]

        # Specify the models you want to save to disk.
        # "G" means "Generator".
        # "D" means "Discriminator".
        self.model_names = ["G", "D"]



