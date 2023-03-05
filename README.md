# Deep Learning Theory and Practice

Hi Friends:

仓库会尝试用尽可能统一的框架去实现一些常见CV任务中的经典网络。目前仓库中实现了部分图像分类的网络，近期会新加入GAN和一些用于Image2Image任务的算法，如去模糊。

我们的目标是：仅用配置文件就能实现非常丰富的网络结构定义。

纯业余时间在写，测试用例无法覆盖完全。 有bug或意见烦请指出、反馈！

--------------------
三方依赖（列出的是已测试版本。向下兼容的高版本也行）：

PyTorch: 1.11.0

pyyaml：0.6.9

ptflops：6.0

scipy: 1.5.1

Pillow: 7.2.0

opencv: 4.5.5

--------------------


## Classification

对于分类任务，可以使用以下两种方法之一来训练模型： 

1. 用IDE去运行trainer_classification.py，仅需在代码中指定config_file即可；
2. 使用如下命令来指定一个config_file来启动训练：


```bash
python trainer_classification.py --config_file your_config_file.yaml
```

关于config_file的写法，仓库中在“classification/configs/”目录下包含了大量针对不同模型写的config，可以参考。此处我们也会做详细说明。

以下方用于训练ResNet56的代码作为例子。

~~~python
Dataset:
  name: "CIFAR10"
  root_path: "Datasets"
  batch_size: 128
  h: 32
  w: 32
Argumentation:
  RandomHorizontalFlip:
    p: 0.5
  RandomCrop:
    size: [32, 32]
    padding: 4
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
Model:
  ResNet:
    block: "ResBlock"
    n_blocks_list: [9, 9, 9]
    stride_list: [1, 2, 2]
    in_channels: 3
    hidden_channels: 16
    kernel_size: 3
    kernel_size_first: 3
    stride_first: 1
    norm: "BatchNorm"
    act: "Relu"
    down_sample: "interpolate"
    bias: True
    use_short_cut: True
    use_maxpool: False
    num_classes: 10
  OPT:
    SGD:
      lr: 0.1
      momentum: 0.9
      weight_decay: 1e-4
  Scheduler:
    MultiStepLR: 
      gamma: 0.1
      milestones: [32000, 48000]
  Init:
    kaiming_normal_:
      mode: "fan_in"
      nonlinearity: "relu"

Train:
  iterations: 64000
  save_freq: 500
  output: "Output/ResNet_56-Layers_CIFAR10_EXP"
  keep_gradients: True
  device: "cuda"
~~~
整个配置文件由四部分构成，Dataset、Argumentation、Model和Train构成。
### Dataset ###
Dataset是与数据相关的配置参数。

有两种方法来指定数据集：

1. 通过关键字name来指定PyTorch内置的数据集。并通过root_path关键字来指定数据集存放的路径。当数据集不存在时，会自动下载到该目录下。目前只支持CIFAR10。

2. 通过关键字trn_path和tst_path来分别指定训练集和测试集所在的根目录。在这种设置下，数据需要按照如下结构准备：

trn_path下包含多个文件夹，每一个文件夹表示不同类别。属于同一类别的图像存放在对应的文件夹下。

例如有一个区分猫狗的数据集。训练集在“cat_dog/”目录下。那么“cat_dog/”目录需要包含一个"cat"文件夹和一个"dog"文件夹。
"cat"文件夹中存放所有用于训练的猫的图像；"dog"文件夹中存放所有用于训练的狗的图像。

tst_path类似处理。

batch_size用来指定批大小。

h、w分别指定输入图像的高和宽。输入图像会首先缩放至尺寸(h, w)。

### Argumentation ###
Argumentation是与数据增强相关的配置参数。
Argumentation下可以写入任何PyTorch支持的增强方法。写法很简单，只需首先写入增强方法的名称（如RandomHorizontalFlip），然后再在该关键字下写入参数（如p: 0.5）。
需要注意，方法名和参数名需要与PyTorch严格一致。
多个增强方法需要注意它们的顺序，由上至下执行。
to_tensor()不用写在此处，会自动加入。

mean和std是用于数据标准化的参数，默认值来自ImageNet。

### Model ###
（未完待续）