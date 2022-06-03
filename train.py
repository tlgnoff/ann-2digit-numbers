from math import sqrt
import sys
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import TrainDataset, ValDataset
from model import OneLayerModel, TwoLayerModel, ThreeLayerModel


def train(model, optimizer, train_dataloader, val_dataloader, epochs, activation_func, device, best_accuracy, layers_idx, func_idx, neurons, learning_rate):
    min_val_loss = 10000
    epochs_no_improve = 0
    n_epochs_stop = 10
    early = False
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
            if correct/total > best_accuracy:
                best_accuracy = correct/total
                print('Best for now: Layers - {}, Function - {}, Neurons - {}, Learning Rate - {}, Accuracy - {}'.format(layers_idx+1, func_idx, neurons, learning_rate, best_accuracy))
                torch.save(model.state_dict(), 'model_state_dict')
            val_loss = val_loss/len(val_dataloader)
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
                if epochs_no_improve == n_epochs_stop:
                    early = True
                    print('Early stopping! Epoch: {}'.format(epoch_idx))
                    print('Training loss: {}.\tValidation loss: {}\tValidation accuracy: {}'.format(train_loss/len(train_dataloader), min_val_loss, correct/total))
                    break
    if not early:
        print('Training loss: {}.\tValidation loss: {}\tValidation accuracy: {}'.format(train_loss/len(train_dataloader), val_loss, correct/total))
    return best_accuracy



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
    best_accuracy = 0.0
    for layers_idx in range(3):
        for func_idx in range(3):
            neurons = 256
            for neuron_idx in range(3):
                if func_idx == 0:
                    activation_func = F.sigmoid
                elif func_idx == 1:
                    activation_func = F.tanh
                else:
                    activation_func = F.relu

                learning_rate = 0.01
                for lr_idx in range(6):
                    if layers_idx == 0:
                        model = OneLayerModel()
                    elif layers_idx == 1:
                        model = TwoLayerModel(neurons)
                    else:
                        model = ThreeLayerModel(neurons)

                    model = model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
                    print('Layers: {},\tActivation func: {},\tNeurons: {},\tLearning rate: {}'.format(layers_idx+1, func_idx, neurons, learning_rate))
                    best_accuracy = train(model, optimizer, train_dataloader, val_dataloader, epochs, activation_func, device, best_accuracy, layers_idx, func_idx, neurons, learning_rate)
                    learning_rate /= sqrt(10)
                if layers_idx == 0:
                    break
                neurons *= 2
            if layers_idx == 0:
                break


if __name__ == '__main__':
    main()






