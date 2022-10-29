import torch
from torchvision import models

model = models.vgg19()
print(model)


# model = torch.hub.load("pytorch/vision:v0.10.0", "vgg19", pretrained=True)
# model.eval()
