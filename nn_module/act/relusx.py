import torch.nn as nn

import register


@register.NAME_TO_ACTS.register("ReLUSX")
class ReLUSX(nn.Module):
    """
    Flip ReLU with x axis.
    This function is designed to deal with the problem when training
    ResNet with subtraction instead of addition for residual connection.
    See ZhiHu for details:
    https://www.zhihu.com/question/433548556/answer/2938153423
    """

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        return -self.act(x)