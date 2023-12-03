# Deep Learning Theory and Practice

## Hello

--------------------
欢迎来到Deep Learning Theory and Practice！

仓库会尝试用尽可能统一的框架去实现一些常见CV任务中的经典网络。我们的目标是：仅用配置文件就能实现非常丰富的网络结构定义，而不依靠大量的代码。

目前仓库的代码框架进行了大量更新，原来老版本请使用Releases中的v0.1版本。老版本的代码不会再更新。

新版本中，新增了大量的wiki来介绍一些框架的基本信息和使用方法，相信能帮助大家更好地理解和使用本仓库代码。

Keep in mind：
* 仓库所有代码是业务时间在写，精力有限，测试用例覆盖不全，可能会有bug，见谅的同时也欢迎指出。
* 欢迎大家向仓库贡献代码或config文件。
* 如果你有新的需求，欢迎提issue或知乎私信，我会尽量满足大家的需求。
* 仓库有wiki，里面写了关于仓库的详细介绍，欢迎大家阅读。
* 如果有帮助，希望给个star，谢谢！

## 如何使用

--------------------

以分类任务为例，要训练某个网络，有两种方式：

* 第一种方式是通过命令行参数指定配置文件路径（将xxx.yaml替换为配置文件路径）:
```shell 
python trainer_classification.py --config_file xxx.yaml
```
例如，当需要训练MobileNet v1时，可以使用：
```shell
python trainer_classification.py --config_file classification/configs/MobileNet_v1/MobileNet_ImageNet_224_EXP.yaml
```

* 第二种方式是将trainer_classification.py中的config_file变量的默认值值指定为配置文件路径，然后直接运行trainer_classification.py即可。

要训练GAN网络类似，只需要将trainer_classification.py替换为trainer_gan.py即可。

## 已支持的网络结构

### 分类网络

> 没打钩的表示利用仓库中的Network和Block能通过配置文件实现的网络，但对应的config文件还没加入到仓库中。所有打钩的在仓库中都有对应配置文件。其目录在：classification/configs/。


- [x] ResNet：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [x] PreActResNet：[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- [x] 减法ResNet：[Resnet的加法变成减法会怎样？](https://www.zhihu.com/question/433548556/answer/2938153423)
- [x] VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [x] MobileNet V1: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [x] MobileNet V2: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [x] MobileNet V3: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [x] ACNet: [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](https://arxiv.org/abs/1908.03930)
- [x] DBBNet: [Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)
- [x] RepVGG: [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
- [x] DiracNet: [DiracNets: Training Very Deep Neural Networks Without Skip-Connections](https://arxiv.org/abs/1706.00388)
- [x] MobileOne: [MobileOne: An Improved One millisecond Mobile Backbone](https://arxiv.org/abs/2206.04040)
- [ ] SqueezeNet: [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
- [ ] Squeeze-and-Excitation Net: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

### GAN

> config目录：gan/configs/

- [x] DCGAN：[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- [x] WGAN：[Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [x] WGAN-GP：[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)


## 三方依赖：

--------------------

列出的是已测试版本:
- [x] Pillow: 9.5.0
- [x] ptflops：0.7
- [x] pyyaml：6.0
- [x] PyTorch: 2.0.0
- [x] torchmetrics: 0.11.4
- [x] scipy: 1.5.1
- [x] opencv-python: 4.5.5

## 仓库适合谁？

--------------------

* **学生**。本仓库代码不是按照特定网络结构case-by-case实现的，所以大家能学习到一些基本的关于框架的概念和设计技巧。
* **想做可视化算法平台的工程师**。仓库能大大降低算法的核心代码量，任何新的算法可能只需要定义一个新的Block和配置文件。
* **算法爱好者**。仓库代码给了详细的代码解释，尤其对于复杂的pipeline，能让大家更好理解代码。


## 更新日志

--------------------

**2023/03/05**

更新GAN。为了框架统一，更改了大量之前代码。首先个人精力，测试用例不足，可能会有bug。欢迎指出。

**2023/12/03**

大更新。使用了更加复杂的设计来统一整体框架。框架的核心概念变为Network和Block。整体框架更加灵活和强大，但配置文件的写法也更加复杂。

## To do list

- [ ] 目标检测框架（进行中）
- [ ] Stable Diffusion
- [ ] ???

## wiki

--------------------

请记得仓库有[wiki](https://github.com/civat/DL-Theory-Practice/wiki)。
