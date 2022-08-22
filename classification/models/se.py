# The code is from https://github.com/DingXiaoH/RepVGG/blob/main/se_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=hidden_channels, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x