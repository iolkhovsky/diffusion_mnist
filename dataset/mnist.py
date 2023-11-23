import lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


class CustomImageDataset(Dataset):
    def __init__(self, classes=10, samples_per_class=8):
        self.classes = classes
        self.samples = samples_per_class

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor(
            [[c for i in range(self.samples)] for c in range(self.classes)]
        ).long()


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='raw_data/mnist', batch_size=128, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.dataset = None

    def setup(self, stage=None):
        self.dataset = MNIST(self.data_dir, train=True, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(CustomImageDataset(classes=10, samples_per_class=8))
