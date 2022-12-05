import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from data import Data

# from vgg19 import VGG19


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
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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


if __name__ == "__main__":
    data = Data()
    path = "./cat.jpg"
    img = Image.open(path)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    tensor = transform(img)
    img_numpy = data.unnormalize(tensor)
    plt.figure()
    plt.imshow(img_numpy)
    # plt.show()

    PATH = "./cifar_net1.pth"
    net = VGG19(10)
    # net = models.vgg19_bn()
    # net.classifier._modules["6"] = nn.Linear(4096, 10)
    net.load_state_dict(torch.load(PATH))
    net.to("mps")

    # image = tensor.unsqueeze(0)
    # output = net(image)
    # probs = torch.nn.functional.softmax(output, dim=1)
    # conf, classes = torch.max(probs, 1)
    # plt.figure()
    # plt.title(
    #     f"Confidence={round(conf.item(), 2)}\nPrediction Label: {data.classes[classes.item()]}"
    # )
    # plt.imshow(img_numpy)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in data.classes}
    total_pred = {classname: 0 for classname in data.classes}

    for images, labels in tqdm(data.testloader):
        images, labels = images.to("mps"), labels.to("mps")
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[data.classes[label]] += 1
            total_pred[data.classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
