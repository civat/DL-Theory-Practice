import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plainnet20_train_loss = torch.load("../../output/PlainNet_20-Layers_CIFAR10/last.pth")["trn_loss"]
    plainnet56_train_loss = torch.load("../../output/PlainNet_56-Layers_CIFAR10/last.pth")["trn_loss"]
    plt.figure(figsize=(12, 8), dpi=80)
    epoch_list = [i + 1 for i in range(len(plainnet20_train_loss))]
    plt.plot(epoch_list, plainnet20_train_loss, color="red", label="training_loss_20")
    plt.plot(epoch_list, plainnet56_train_loss, color="blue", label="training_loss_56")
    plt.xlabel(f"iterations x500")
    plt.ylabel("training loss")
    plt.legend(loc="upper right")
    plt.savefig("../../output/ResNet/PlainNet_train_loss.jpg")
    plt.close()
