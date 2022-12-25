import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class GANDataset(Dataset):

    def __init__(self, data_path, vec_dim, transforms=None):
        super(GANDataset, self).__init__()
        self.vec_dim = vec_dim
        self.transforms = transforms
        self.images = []

        files = os.listdir(data_path)
        for file in files:
            self.images.append(os.path.join(data_path, file))

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image) if self.transforms is not None else image
        vec_rand = torch.normal(mean=0.0, std=1.0, size=(1, self.vec_dim)).squeeze(0)

        return {
            "real": image,
            "vec": vec_rand
        }

    def __len__(self):
        return len(self.images)
