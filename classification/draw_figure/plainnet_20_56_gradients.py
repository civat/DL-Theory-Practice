import os
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print(f"The current path is: {os.getcwd()}")
    plainnet20_state = torch.load("Output/PlainNet_20-Layers_CIFAR10/last.pth")["gradients"]
    plainnet56_state = torch.load("Output/PlainNet_56-Layers_CIFAR10/last.pth")["gradients"]

    # Change the layer number to show gradients of different layers.
    layer = "convs.0"
    plainnet20_gradients = plainnet20_state[layer]
    plainnet56_gradients = plainnet56_state[layer]

    plt.figure(figsize=(12, 8), dpi=200)
    epoch_list = [i + 1 for i in range(len(plainnet20_gradients))]
    plt.plot(epoch_list, plainnet20_gradients, color="red", label="gradient_20")
    plt.plot(epoch_list, plainnet56_gradients, color="blue", label="gradient_56")
    plt.xlabel(f"iterations x500")
    plt.ylabel("gradients")
    plt.legend(loc="upper right")

    if not os.path.exists("Output/ResNet_Figures"):
        os.makedirs("Output/ResNet_Figures")
    plt.savefig(f"Output/ResNet_Figures/PlainNet_train_gradient_{layer}.jpg")
    plt.close()
