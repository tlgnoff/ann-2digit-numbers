import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import ValDataset
from model import OneLayerModel


def main():
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])
    val_dataset = ValDataset('data', 'train', transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=4)
    activation_func = F.relu
    model = OneLayerModel()
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            outputs = model(images, activation_func)
            _, predicted = torch.max(outputs.data, 1)
            loss = F.nll_loss(outputs, labels)
            print('Calculated loss: {}'.format(loss.item()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('Calculated accuracy: {}'.format(correct/total))
            exit()




if __name__ == '__main__':
    main()






