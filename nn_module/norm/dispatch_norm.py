import torch.nn as nn

import register


class DispatchNorm(nn.Module):
    """
    This class is used to dispatch Norm method.
    User can always use this Norm in network design.
    We separate the implementation of user-designed norms
    and PyTorch official norms by following way:
      1) for PyTorch official norms, we use if-else to dispatch;
      2) for user-designed norms, we use polymorphism.

    Note: This implementation can be simplified once the Python
          support singledispatch on Class, not only on Class Instance.
    """

    def __init__(self, norm, **kwargs):
        super().__init__()
        self.norm = norm
        self.kwargs = kwargs
        self.norm_layer = self._init_norm()

    def _init_norm(self):
        if isinstance(self.norm, nn.BatchNorm2d):
            return nn.BatchNorm2d(num_features=self.kwargs["num_features"])
        else:
            return self.norm(**self.kwargs)

    def forward(self, x):
        return self.norm_layer(x)


@register.NAME_TO_NORMS.register("IdentityNorm")
class IdentityNorm(nn.Module):
    """
    Identity norm.
    This is generally used as "Identity Norm" in network for
    convenient implementation of "no normalization".
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x