import torch.nn as nn

import register


@register.NAME_TO_ACTS.register('ReLUSX')
class ReLUSX(nn.Module):

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        return -self.act(x)