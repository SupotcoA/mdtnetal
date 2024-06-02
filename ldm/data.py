import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# print("torchvision.transoforms.v2 not available, using v1 instead")
import cv2
import numpy as np
import os


def get_all_image_paths(directory):
    image_extensions = ('.jpg', '.png')
    image_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                absolute_path = os.path.join(root, file)
                image_paths.append(absolute_path)
    return image_paths


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, label=0):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.label = label

        # Gather image paths and labels for all classes
        self.image_paths = get_all_image_paths(data_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.ascontiguousarray(cv2.imread(image_path)[:, :, ::-1])  # Ensure RGB format

        image = self.transform(image)

        return image, self.label


class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images / 4.5  ### / 4.5 see networks.LatentDiffusion.decode
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


class InfiniteDataLoader:

    def __init__(self, *args, **kwargs):
        self.loader = DataLoader(*args, **kwargs)

    def __iter__(self):
        while True:
            for data in self.loader:
                yield data


@torch.no_grad()
def build_dataset_img(model, data_config):
    dataset2label = {name: i for i, name in enumerate(data_config['dataset_names'])}

    for name, i in dataset2label.items():
        if name in ['afhq', 'fa', 'animestyle']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(data_config['image_size'], antialias=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        print("processing", name)
        data_dir = data_config['data_paths'][name]
        dataset = ImageDataset(data_dir,
                               transform=transform,
                               label=i
                               )
        # Create data loader
        data_loader = DataLoader(dataset,
                                 batch_size=data_config['ae_batch_size'],
                                 shuffle=False,
                                 num_workers=4)

        c = 0
        x, cls = None, None
        for images, labels in data_loader:
            c += 1
            if c % 1000 == 0:
                print(f"encoding {c}th batch.")
            images = images.to(model.device)
            x_ = model.encode(images)
            if x is None:
                x = x_.cpu()
                cls = labels
            else:
                x = torch.cat((x, x_.cpu()), dim=0)
                cls = torch.cat((cls, labels), dim=0)
        print(f"x shape: {x.shape}, cls shape: {cls.shape}")
        torch.save(x, os.path.join(data_config['enc_path'], f'{name}_x.pt'))
        torch.save(cls, os.path.join(data_config['enc_path'], f'{name}_cls.pt'))


@torch.no_grad()
def build_cached_dataset(data_config):
    x = []
    cls = []
    for name in data_config['dataset_names']:
        if name in data_config['ignored_dataset']:
            continue
        x.append(torch.load(os.path.join(data_config['enc_inp_path'], f'{name}_x.pt')))
        cls.append(torch.load(os.path.join(data_config['enc_inp_path'], f'{name}_cls.pt')))
    x = torch.cat(x, dim=0)
    cls = torch.cat(cls, dim=0)
    print(f"x shape: {x.shape}, cls shape: {cls.shape}")

    s = x.shape[0]
    select_every_n = round(1 / (1 - data_config['split']))
    is_test = torch.arange(0, s) % select_every_n == 0
    train_data = TensorDataset(x[~is_test], cls[~is_test])
    print("train length", len(train_data))
    test_data = TensorDataset(x[is_test], cls[is_test])
    print("test length", len(test_data))
    train_data_loader = InfiniteDataLoader(train_data,
                                           batch_size=data_config['batch_size'],
                                           shuffle=True,
                                           num_workers=4)
    test_data_loader = DataLoader(test_data,
                                  batch_size=data_config['batch_size'],
                                  shuffle=False,
                                  num_workers=4)
    return train_data_loader, test_data_loader
