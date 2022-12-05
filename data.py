import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


class Data:
    def __init__(self) -> None:
        print("loading data")
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                # transforms.RandomHorizontalFlip(p=0.5),
                # torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.batch_size = 16

        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.train_transform
        )

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.val_transform
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
        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                # transforms.RandomGrayscale(p=0.3),
            ]
        )
        # processing
        input_tensor = preprocess(tensor)

        # input_batch = input_tensor.unsqueeze(0)
        # print(input_batch)
        return input_tensor
