from pathlib import Path
from typing import Optional, List, Tuple, Callable

import numpy as np
import pytorch_lightning as pl
import requests
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

from config import *

class DSpritesDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        return_latents: bool,
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.filepath = self.data_dir / FILENAME_DSPRITES
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.return_latents = return_latents

        self.data_dir.mkdir(parents = True, exist_ok = True)

        if not self.filepath.exists():
            print(f"DSprites not found. Downloading from: {URL_DSPRITES}...")
            self._download()

        with np.load(self.filepath, allow_pickle = True, encoding = 'bytes') as dataset_zip:
            self.imgs = dataset_zip['imgs']
            self.latents_values = dataset_zip['latents_values']
        
    def _download(self) -> None:
        try:
            with requests.get(URL_DSPRITES, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))

                with open(self.filepath, 'wb') as file, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc="Downloading dSprites"
                ) as progress_bar:
                    for data in response.iter_content(chunk_size=1024):
                        file.write(data)
                        progress_bar.update(len(data))
        except requests.exceptions.RequestException as e:
            print(f"DSprites not downloaded. Error: {e}")
            raise

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        image = self.imgs[idx].astype(np.float32)
        
        if self.transform:
            image = self.transform(image)

        if self.return_latents:
            latents = torch.tensor(self.latents_values[idx], dtype = torch.float32)
            return image, latents
        
        return image


class DSpritesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        train_val_test_split: List[float]
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.dsprites_train: Optional[Dataset] = None
        self.dsprites_val: Optional[Dataset] = None
        self.dsprites_test: Optional[Dataset] = None

    def prepare_data(self) -> None:

        DSpritesDataset(self.data_dir, return_latents = True)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dsprites_train is None:
            dsprites_full = DSpritesDataset(
                data_dir=self.data_dir,
                transform=self.transform,
                return_latents=True
            )
            total_len = len(dsprites_full)
            train_len = int(self.train_val_test_split[0] * total_len)
            val_len = int(self.train_val_test_split[1] * total_len)
            test_len = total_len - train_len - val_len

            self.dsprites_train, self.dsprites_val, self.dsprites_test = random_split(
                dsprites_full,
                [train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(SEED)
            )
            print(f"Dataset: Train = {len(self.dsprites_train)}, Val = {len(self.dsprites_val)}, Test = {len(self.dsprites_test)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dsprites_train,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            pin_memory = True,
            persistent_workers = True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dsprites_val,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = True,
            persistent_workers = True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dsprites_test,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = True,
            persistent_workers = True
        )

