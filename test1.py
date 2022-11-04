import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


# if gpu is avalible then use gpu
device = "mps" if torch.backends.mps.is_available() else "cpu"


class Data:
    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.batch_size = 16

        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )

        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    def unnormalize(self, img) -> None:
        """show images"""
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()
        npimg = np.transpose(npimg, (1, 2, 0))

        return npimg

    def augmentation(self, tensor):
        """
        Read torch tensor, do argumentation then return tensor
        """

        # define preprocessing step
        degree = random.choices([0, 10, 20, 30])[0]
        shearing_degree = random.choices([0, 10, 15, 20], weights=[60, 20, 10, 10])[0]
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomAffine(
                    degree,
                    translate=None,
                    scale=None,
                    shear=((0, 0, 0, shearing_degree)),
                ),
                transforms.RandomGrayscale(p=0.3),
            ]
        )
        # processing
        input_tensor = preprocess(tensor)

        # input_batch = input_tensor.unsqueeze(0)
        # print(input_batch)
        return input_tensor


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
    # plt.figure()
    # for index, img in enumerate(images, start=1):
    #     npimg = data.unnormalize(img)
    #     plt.subplot(3, 3, index)
    #     plt.title(f"{data.classes[labels[index - 1]]:5s}", fontsize=10)
    #     plt.imshow(npimg)
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()

    """
    Q5.3
    """
    plt.figure()
    origin = data.unnormalize(images[0])
    plt.subplot(2, 3, 2)
    plt.title("origin")
    plt.imshow(origin)
    plt.xticks([])
    plt.yticks([])
    for i in range(4, 7):
        inputs = data.augmentation(images)
        tensor = inputs[0]
        trans = data.unnormalize(tensor)
        plt.subplot(2, 3, i)
        plt.title(f"trans{i-3}")
        plt.imshow(trans)
        plt.xticks([])
        plt.yticks([])
    plt.show()
