import os
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print(f"The current path is: {os.getcwd()}")
    plainnet20_train_loss = torch.load("Output/PlainNet_20-Layers_CIFAR10/last.pth")["trn_loss"]
    plainnet56_train_loss = torch.load("Output/PlainNet_56-Layers_CIFAR10/last.pth")["trn_loss"]
    plt.figure(figsize=(12, 8), dpi=80)
    epoch_list = [i + 1 for i in range(len(plainnet20_train_loss))]
    plt.plot(epoch_list, plainnet20_train_loss, color="red", label="training_loss_20")
    plt.plot(epoch_list, plainnet56_train_loss, color="blue", label="training_loss_56")
    plt.xlabel(f"iterations x500")
    plt.ylabel("training loss")
    plt.legend(loc="upper right")

    if not os.path.exists("Output/ResNet_Figures"):
        os.makedirs("Output/ResNet_Figures")
    plt.savefig("Output/ResNet_Figures/PlainNet_train_loss.jpg")
    plt.close()
