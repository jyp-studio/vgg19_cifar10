import torch.nn as nn


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
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, classes),
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
        layers += [nn.AdaptiveAvgPool2d(7)]
        return nn.Sequential(*layers)
