import os
from torch.utils.data import Dataset
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, dataset_path, split, transforms):
        images_path = os.path.join(dataset_path, split)
        self.data = []
        with open(os.path.join(images_path, 'labels.txt'), 'r') as f:
            for i in range(9000):
                line = next(f).strip()
                image_name, label = line.split()
                image_path = os.path.join(images_path, image_name)
                label = int(label)
                self.data.append((image_path, label))

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item][0]
        label = self.data[item][1]
        image = Image.open(image_path)
        image = self.transforms(image)

        return image, label


class ValDataset(Dataset):
    def __init__(self, dataset_path, split, transforms):
        images_path = os.path.join(dataset_path, split)
        self.data = []
        with open(os.path.join(images_path, 'labels.txt'), 'r') as f:
            for i in range(9000):
                next(f)
            for i in range(1000):
                line = next(f).strip()
                image_name, label = line.split()
                image_path = os.path.join(images_path, image_name)
                label = int(label)
                self.data.append((image_path, label))
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item][0]
        label = self.data[item][1]
        image = Image.open(image_path)
        image = self.transforms(image)

        return image, label


class TestDataset(Dataset):
    def __init__(self, dataset_path, split, transforms):
        images_path = os.path.join(dataset_path, split)
        self.data = []
        with open(os.path.join(images_path, 'labels.txt'), 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                image_name = line
                image_path = os.path.join(images_path, image_name)
                self.data.append((image_path, image_name))

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item][0]
        image_name = self.data[item][1]
        image = Image.open(image_path)
        image = self.transforms(image)

        return image, image_name


