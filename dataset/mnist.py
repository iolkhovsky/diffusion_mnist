import lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


class DatsetPlug(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor(idx)


class ToTensorNoScaling(transforms.ToTensor):
    def __call__(self, x):
        tensor = super().__call__(x)
        return tensor * 255.0


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='raw_data/mnist', batch_size=128, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            ToTensorNoScaling(),
        ])
        self.dataset = None

    def setup(self, stage=None):
        self.dataset = MNIST(self.data_dir, train=True, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(DatsetPlug())
