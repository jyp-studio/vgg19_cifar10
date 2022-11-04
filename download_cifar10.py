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

    def augmentation(self, tensor):
        """
        Read torch tensor, do argumentation then return tensor
        """

        # define preprocessing step
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.RandomGrayscale(p=0.3),
            ]
        )
        # processing
        input_tensor = preprocess(tensor)

        # input_batch = input_tensor.unsqueeze(0)
        # print(input_batch)
        return input_tensor


class VGG19(nn.Module):
    def __init__(self, classes):
        super(VGG19, self).__init__()
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
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x.view(x.size(0), -1))
        x = self.classifier(x)

        return x

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
        layers += [nn.AdaptiveAvgPool2d(4)]
        return nn.Sequential(*layers)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 16, 3, 1, padding=1
        )  # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
        self.conv2 = nn.Conv2d(
            16, 32, 3, 1, padding=1
        )  # We double the feature maps for every conv layer as in pratice it is really good.
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(
            4 * 4 * 64, 500
        )  # I/p image size is 32*32, after 3 MaxPooling layers it reduces to 4*4 and 64 because our last conv layer has 64 outputs. Output nodes is 500
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(
            500, 10
        )  # output nodes are 10 because our dataset have 10 different categories

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply relu to each output of conv layer.
        x = F.max_pool2d(x, 2, 2)  # Max pooling layer with kernal of 2 and stride of 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(
            -1, 4 * 4 * 64
        )  # flatten our images to 1D to input it to the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(
            x
        )  # Applying dropout b/t layers which exchange highest parameters. This is a good practice
        x = self.fc2(x)
        return x


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
    Q5.2
    """
    model = VGG19(10)
    summary(model, (3, 32, 32), device="cpu")
    model.to(device=device)
    """
    Q5.3
    """
    # plt.figure()
    # origin = data.unnormalize(images[0])
    # plt.subplot(2, 3, 2)
    # plt.title("origin")
    # plt.imshow(origin)
    # plt.xticks([])
    # plt.yticks([])
    # for i in range(4, 7):
    #     tensor = data.augmentation(images[0])
    #     trans = data.unnormalize(tensor)
    #     plt.subplot(2, 3, i)
    #     plt.title(f"trans{i-3}")
    #     plt.imshow(trans)
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()

    """
    Q5.4
    """
    # model = LeNet().to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 2
    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    for e in range(epochs):  # training our model, put input according to every batch.

        running_loss = 0.0
        running_corrects = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0.0

        for inputs, labels in data.trainloader:
            model.train()
            inputs = data.augmentation(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # every batch of 100 images are put as an input.
            outputs = model(inputs)
            # Calc loss after each batch i/p by comparing it to actual labels.
            loss = criterion(outputs, labels)
            # setting the initial gradient to 0
            optimizer.zero_grad()
            # backpropagating the loss
            loss.backward()
            # updating the weights and bias values for every single step.
            optimizer.step()

            # taking the highest value of prediction.
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            # calculating te accuracy by taking the sum of all the correct predictions in a batch.
            running_corrects += torch.sum(preds == labels.data)

        else:
            # we do not need gradient for validation.
            with torch.no_grad():
                for val_inputs, val_labels in data.testloader:
                    model.eval()
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)

                    _, val_preds = torch.max(val_outputs, 1)
                    val_running_loss += val_loss.item()
                    val_running_corrects += torch.sum(val_preds == val_labels.data)

            epoch_loss = running_loss / len(data.trainloader)  # loss per epoch
            epoch_acc = running_corrects.float() / len(
                data.trainloader
            )  # accuracy per epoch
            running_loss_history.append(epoch_loss)  # appending for displaying
            print(epoch_acc.cpu())
            running_corrects_history.append(epoch_acc.item())

            val_epoch_loss = val_running_loss / len(data.testloader)
            val_epoch_acc = val_running_corrects.float() / len(data.testloader)
            val_running_loss_history.append(val_epoch_loss)
            val_running_corrects_history.append(val_epoch_acc.item())
            print("epoch:", (e + 1))
            print(
                "training loss: {:.4f}, acc {:.4f} ".format(
                    epoch_loss, epoch_acc.item()
                )
            )
            print(
                "validation loss: {:.4f}, validation acc {:.4f} ".format(
                    val_epoch_loss, val_epoch_acc.item()
                )
            )

    PATH = "./cifar_net2.pth"
    torch.save(model.state_dict(), PATH)

    # loss
    plt.style.use("ggplot")
    plt.plot(running_loss_history, label="training loss")
    plt.plot(val_running_loss_history, label="validation loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()

    # accuracy
    plt.style.use("ggplot")
    plt.plot(running_corrects_history, label="training accuracy")
    plt.plot(val_running_corrects_history, label="validation accuracy")
    plt.legend()
    plt.savefig("accuracy.png")
    plt.show()

    """
    dataiter = iter(data.testloader)
    images, labels = next(dataiter)

    # print images
    plt.figure()
    for index, img in enumerate(images, start=1):
        test = data.unnormalize(img)
        plt.subplot(3, 3, index)
        plt.title(f"{data.classes[labels[index - 1]]:5s}", fontsize=10)
        plt.imshow(test)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    """
