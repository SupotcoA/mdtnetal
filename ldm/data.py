import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# print("torchvision.transoforms.v2 not available, using v1 instead")
import cv2
import numpy as np
import os
import glob


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, label=0):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.label = label

        # Gather image paths and labels for all classes
        self.image_paths = []
        for extension in ['*.jpg', '*.png']:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, extension)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.ascontiguousarray(cv2.imread(image_path)[:, :, ::-1])  # Ensure RGB format

        image = self.transform(image)

        return image, self.label


class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
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

    for i, name in dataset2label:
        if name in ['afhq', 'fa', 'animestyle']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(data_config['image_size']),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        print("processing", name)
        data_dir = data_config['data_paths']['name']
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
    x = torch.load(data_config['x_path'])
    cls = torch.load(data_config['cls_path'])
    print(f"x shape: {x.shape}, cls shape: {cls.shape}")
    # assert x.shape[0] == 15000
    s = x.shape[0]
    split = int(s * data_config['split'])
    perm_idx = torch.randperm(x.shape[0])
    train_idx = perm_idx[:split]
    test_idx = perm_idx[split:]
    train_data = TensorDataset(x[train_idx], cls[train_idx])
    test_data = TensorDataset(x[test_idx], cls[test_idx])
    train_data_loader = InfiniteDataLoader(train_data,
                                           batch_size=data_config['batch_size'],
                                           shuffle=True,
                                           num_workers=4)
    test_data_loader = DataLoader(test_data,
                                  batch_size=data_config['batch_size'],
                                  shuffle=True,
                                  num_workers=4)
    return train_data_loader, test_data_loader
