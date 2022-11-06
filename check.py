from data import Data, VGG19
import torch
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    data = Data()
    path = "./cat.jpg"
    img = Image.open(path)
    tensor = data.transform(img)
    img_numpy = data.unnormalize(tensor)
    plt.figure()
    plt.imshow(img_numpy)
    # plt.show()

    PATH = "./cifar_net1.pth"
    net = VGG19(10)
    net.load_state_dict(torch.load(PATH))

    image = tensor.unsqueeze(0)
    output = net(image)
    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    plt.figure()
    plt.title(
        f"Confidence={round(conf.item(), 2)}\nPrediction Label: {data.classes[classes.item()]}"
    )
    plt.imshow(img_numpy)
    plt.xticks([])
    plt.yticks([])
    plt.show()
