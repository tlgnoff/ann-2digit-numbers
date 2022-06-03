from math import sqrt

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import TestDataset
from model import ThreeLayerModel
import sys


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'cuda':
            use_cuda = True
        else:
            use_cuda = False
    else:
        use_cuda = False
    device = torch.device('cuda' if use_cuda else 'cpu')
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])
    test_dataset = TestDataset('data', 'test', transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=512, num_workers=4)
    activation_func = F.sigmoid
    model = ThreeLayerModel(1024)
    model.load_state_dict(torch.load('model_state_dict'))
    model.to(device)
    model.eval()
    f = open("prediction.txt", "w+")
    with torch.no_grad():
        for images, image_name in test_dataloader:
            images = images.to(device)
            outputs = model(images, activation_func)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(image_name)):
                f.write('{} {}\n'.format(image_name[i], predicted[i].item()))
    f.close()




if __name__ == '__main__':
    main()






