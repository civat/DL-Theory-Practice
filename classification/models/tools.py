from classification import nnblock
from classification.models import acb
from classification.models import dbb
from classification.models import repvgg


def get_conv(configs):
    conv_name = configs["conv"] if "conv" in configs else "Conv2d"
    assert conv_name in convs.keys()
    conv = convs[conv_name]
    conv = conv.get_conv(configs)
    return conv


convs = {
    "Conv2d": nnblock.Conv2d,
    "ACBlock": acb.ACBlock,
    "DBBlock": dbb.DiverseBranchBlock,
    "RepVGGBlock": repvgg.RepVGGBlock,
}
