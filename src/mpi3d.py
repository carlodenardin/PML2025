from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pytorch_lightning as pl
import requests
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from config import *

class MPI3DDataset(Dataset):
    def __init__(self, data_dir: str, return_latents: bool = True):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.filepath = self.data_dir / FILENAME_MPI3D
        self.return_latents = return_latents

        self.data_dir.mkdir(parents=True, exist_ok=True)

        if not self.filepath.exists():
            print(f"File MPI3D non trovato. Inizio download da: {URL_MPI3D}...")
            self._download()

        with np.load(self.filepath) as dataset_zip:
            images_uint8 = dataset_zip['images']
            self.images = images_uint8.astype(np.float32) / 255.0
            self.latents_values = dataset_zip['latents_values']
        self.lat_sizes = np.array([len(np.unique(self.latents_values[:, i])) for i in range(self.latents_values.shape[1])])

    def _download(self) -> None:
        try:
            with requests.get(URL_MPI3D, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                with open(self.filepath, 'wb') as file, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc="Downloading MPI3D"
                ) as progress_bar:
                    for data in response.iter_content(chunk_size=8192):
                        file.write(data)
                        progress_bar.update(len(data))
        except requests.exceptions.RequestException as e:
            print(f"Download di MPI3D fallito. Errore: {e}")
            if self.filepath.exists(): self.filepath.unlink()
            raise

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        if self.return_latents:
            latents = torch.tensor(self.latents_values[idx], dtype=torch.float32)
            return image, latents
        return image

class MPI3DDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, train_val_test_split: List[float]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.mpi3d_full: Optional[Dataset] = None
        self.mpi3d_train: Optional[Dataset] = None
        self.mpi3d_val: Optional[Dataset] = None
        self.mpi3d_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        MPI3DDataset(self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.mpi3d_train is None:
            self.mpi3d_full = MPI3DDataset(self.data_dir)
            total_len = len(self.mpi3d_full)
            train_len, val_len = int(self.train_val_test_split[0] * total_len), int(self.train_val_test_split[1] * total_len)
            test_len = total_len - train_len - val_len
            self.mpi3d_train, self.mpi3d_val, self.mpi3d_test = random_split(
                self.mpi3d_full, [train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(SEED)
            )
            print(f"Dataset MPI3D suddiviso: Train={len(self.mpi3d_train)}, Val={len(self.mpi3d_val)}, Test={len(self.mpi3d_test)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mpi3d_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mpi3d_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mpi3d_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
