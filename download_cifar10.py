import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# if gpu is avalible then use gpu
device = "mps" if torch.backends.mps.is_available() else "cpu"


class Data:
    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.batch_size = 4

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

    def imshow_1(self, imgs, labels) -> None:
        plt.figure()
        for index, img in enumerate(imgs, start=1):
            npimg = self.unnormalize(img)
            plt.subplot(3, 3, index)
            plt.title(f"{self.classes[labels[index - 1]]:5s}", fontsize=10)
            plt.imshow(npimg)
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def argumentation(self, tensor):
        """
        Read torch tensor, do argumentation then return tensor
        """
        # change the img from tensor to image
        print("tenosr")
        print(tensor)
        trans2Img = transforms.ToPILImage()
        img = trans2Img(tensor)

        # define preprocessing step
        preprocess = transforms.Compose(
            [
                transforms.Resize(32),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomGrayscale(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # processing
        input_tensor = preprocess(img)
        print("input_tensor")
        print(input_tensor)

        # input_batch = input_tensor.unsqueeze(0)
        # print(input_batch)
        return input_tensor

    def convert_cifar10(self, t, pil):
        """Function to convert a cifar10 image tensor (already normalized)
        onto a plotable image.

        :param t: image tensor of size (3,32,23)
        :type t: torch.Tensor
        :param pil: output is of size (3,32,32) if True, else (32,32,3)
        :type pil: bool
        """
        im = torch.Tensor.clone(t)
        # approximate unnormalization
        im[0] = im[0] * 0.229 + 0.485
        im[1] = im[1] * 0.224 + 0.456
        im[2] = im[2] * 0.225 + 0.406
        if not pil:
            im = im.numpy()
            im = np.transpose(im, (1, 2, 0))
        return im


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self._make_layers(
            [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                512,
                "M",
            ]
        )
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


model_vgg19 = models.vgg19()


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
    print(images[0])
    t = transforms.ToPILImage()
    a = t(images[0])
    a.show()
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    b = t(a)
    print(b)

    # show images
    # data.imshow_1(images, labels)

    """
    Q5.2
    """
    tensor = data.argumentation(images[0])
    t = transforms.Resize(32)
    tensor1 = t(tensor)
    original_img = data.convert_cifar10(tensor1, pil=False)
    plt.imshow(original_img)
    print("origin")
    print(original_img)
    plt.show()
    second = data.convert_cifar10(images[0], pil=False)
    plt.imshow(second)
    print("second")
    print(second)
    plt.show()
