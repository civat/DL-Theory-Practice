import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DatasetCls(Dataset):
    """
    Dataset for classification task.
    The root dictionary should be organized in
    the following structure:
    root---sub_dic_1---img1.*
                    ---img2.*
                    ---...
        ---sub_dic_2---img1.*
                    ---img2.*
                    ---...
        ---...
    """

    def __init__(self, root_path, transforms=None):
        super(DatasetCls, self).__init__()
        self.root_path = root_path
        self.transforms = transforms
        self.images = []
        self.labels = []
        self.label_names = []

        sub_dics = os.listdir(root_path)
        for i, sub_dic in enumerate(sub_dics):
            self.label_names.append(sub_dic)
            files = os.listdir(os.path.join(root_path, sub_dic))
            for file in files:
                self.images.append(os.path.join(root_path, sub_dic, file))
                self.labels.append(i)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = self.labels[index]
        image = self.transforms(image) if self.transforms is not None else image
        return image, torch.from_numpy(np.array(label)).long()

    def __len__(self):
        return len(self.images)
