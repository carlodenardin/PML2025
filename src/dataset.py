import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl

class CelebADataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, img_size = 64, batch_size: int = 32, num_workers: int = None):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else (os.cpu_count() // 2 or 1)
        self.transform = transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        try:
            datasets.CelebA(root = self.data_dir, split = 'all', download = True)
        except Exception as e:
            print(f"Error downloading CelebA dataset: {e}")
            raise

    def setup(self, stage = None):
        self.train_dataset = datasets.CelebA(
            root = self.data_dir,
            split = 'train',
            transform = self.transform,
            download = True
        )
        
        self.val_dataset = datasets.CelebA(
            root = self.data_dir,
            split = 'valid',
            transform= self.transform,
            download = True
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False, num_workers=self.num_workers)