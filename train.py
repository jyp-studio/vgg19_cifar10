import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


def train(model, device, data, epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    for e in range(epochs):  # training our model, put input according to every batch.
        print(f"Epoch: {e+1}")

        running_total = 0
        running_loss = 0.0
        running_corrects = 0.0
        val_total = 0
        val_running_loss = 0.0
        val_running_corrects = 0.0

        model.train()
        for inputs, labels in tqdm(data.trainloader, desc=f"Training"):
            # inputs = data.augmentation(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # setting the initial gradient to 0
            optimizer.zero_grad()
            # every batch of 100 images are put as an input.
            outputs = model(inputs)
            # Calc loss after each batch i/p by comparing it to actual labels.
            loss = criterion(outputs, labels)
            # backpropagating the loss
            loss.backward()
            # updating the weights and bias values for every single step.
            optimizer.step()

            # taking the highest value of prediction.
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()

            # calculating te accuracy by taking the sum of all the correct predictions in a batch.
            running_total += labels.size(0)
            running_corrects += (preds == labels).sum().item()

        # else:
        # we do not need gradient for validation.
        model.eval()
        with torch.no_grad():
            for val_inputs, val_labels in tqdm(data.testloader, desc=f"Validating"):
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_total += labels.size(0)
                val_running_corrects += (val_preds == val_labels).sum().item()

        # loss per epoch
        epoch_loss = running_loss / running_total
        # accuracy per epoch
        epoch_acc = running_corrects / running_total
        # appending for displaying
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc.item())

        val_epoch_loss = val_running_loss / val_total
        val_epoch_acc = val_running_corrects / val_total
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc.item())

        print(len(running_total), len(data.trainloader))
        print(len(val_total), len(data.testloader))
        print(f"training   loss: {epoch_loss:.4f}, acc: {epoch_acc.item():.4f}")
        print(f"validation loss: {val_epoch_loss:.4f}, acc: {val_epoch_acc.item():.4f}")
        print()

    PATH = "./igame_net.pth"
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
