import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary

from data import Data
from vgg19 import VGG19
from train import train


# if gpu is avalible then use gpu
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)


if __name__ == "__main__":
    data = Data()

    """
    Q5.1
    """
    # get some random training images
    dataiter = iter(
        torch.utils.data.DataLoader(
            data.trainset, batch_size=9, shuffle=True, num_workers=0
        )
    )
    images, labels = next(dataiter)

    # show images
    plt.figure()
    for index, img in enumerate(images, start=1):
        npimg = data.unnormalize(img)
        plt.subplot(3, 3, index)
        plt.title(f"{data.classes[labels[index - 1]]:5s}", fontsize=10)
        plt.imshow(npimg)
        # plt.xticks([])
        # plt.yticks([])
    plt.show()

    """
    Q5.2
    """
    model = VGG19(10)
    summary(model, (3, 32, 32), device="cpu")
    model.to(device=device)
    # model = models.vgg19_bn()
    # model.classifier._modules["6"] = nn.Linear(4096, 10)
    # summary(model, (3, 32, 32))
    # model.to(device)
    """
    Q5.3
    """
    plt.figure()
    origin = data.unnormalize(images[0])
    plt.subplot(2, 3, 2)
    plt.title("origin")
    plt.imshow(origin)
    # plt.xticks([])
    # plt.yticks([])
    for i in range(4, 7):
        tensor = data.augmentation(images[0])
        trans = data.unnormalize(tensor)
        plt.subplot(2, 3, i)
        plt.title(f"trans{i-3}")
        plt.imshow(trans)
        # plt.xticks([])
        # plt.yticks([])
    plt.show()

    """
    Q5.4
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 50
    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    # start train
    train(model, device, data)

    PATH = "./cifar_net2.pth"
    torch.save(model.state_dict(), PATH)

    # loss
    plt.style.use("ggplot")
    plt.plot(running_loss_history, label="training loss")
    plt.plot(val_running_loss_history, label="validation loss")
    plt.legend()
    plt.title("loss")
    plt.savefig("loss.png")
    plt.show()

    # accuracy
    plt.style.use("ggplot")
    plt.plot(running_corrects_history, label="training accuracy")
    plt.plot(val_running_corrects_history, label="validation accuracy")
    plt.legend()
    plt.title("accuracy")
    plt.savefig("accuracy.png")
    plt.show()
