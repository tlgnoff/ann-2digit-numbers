import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import TrainDataset, ValDataset
from model import ThreeLayerModel
import matplotlib.pyplot as plt
import numpy as np
import sys


def train(model, optimizer, train_dataloader, val_dataloader, epochs, activation_func, device):
    training_loss = []
    validation_loss = []
    for epoch_idx in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images, activation_func)
            loss = F.nll_loss(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        training_loss.append(train_loss/len(train_dataloader))
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for val_images, val_labels in val_dataloader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                val_pred = model(val_images, activation_func)
                val_loss += (F.nll_loss(val_pred, val_labels)).item()
                _, predicted = torch.max(val_pred.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
            val_loss = val_loss/len(val_dataloader)
            validation_loss.append(val_loss)
    plt.plot(np.linspace(1, epochs, epochs), training_loss, label="train_loss")
    plt.plot(np.linspace(1, epochs, epochs), validation_loss, label="val_loss")
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'cuda':
            use_cuda = True
        else:
            use_cuda = False
    else:
        use_cuda = False
    device = torch.device('cuda' if use_cuda else 'cpu')
    epochs = 50
    torch.manual_seed(1234)

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = TrainDataset('data', 'train', transforms)
    val_dataset = ValDataset('data', 'train', transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True, num_workers=4)
    model = ThreeLayerModel(1024)
    model = model.to(device)
    learning_rate = 0.001
    activation_func = F.sigmoid
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    train(model, optimizer, train_dataloader, val_dataloader, epochs, activation_func, device)


if __name__ == '__main__':
    main()






