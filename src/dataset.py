import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split # Aggiunto random_split
from torchvision import transforms
import requests
from tqdm import tqdm
import pytorch_lightning as pl

DSPRITES_URL = "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
DSPRITES_FILENAME = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

class DSpritesDataset(Dataset):
    def __init__(self, root_dir="data/dsprites", download=True, transform=None):
        self.root_dir = root_dir
        self.filepath = os.path.join(self.root_dir, DSPRITES_FILENAME)
        self.transform = transform if transform is not None else transforms.ToTensor()

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        if download and not os.path.exists(self.filepath):
            self._download()

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset not found at {self.filepath}. Please ensure it's downloaded.")

        dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        # metadata = dataset_zip['metadata'][()] # Se ti servissero i metadati

    def _download(self):
        print(f"Downloading dSprites dataset to {self.filepath}...")
        response = requests.get(DSPRITES_URL, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024

        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(self.filepath, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong during download")
        else:
            print("Download complete.")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = self.imgs[idx].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image

class DSpritesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/dsprites", batch_size: int = 64, num_workers: int = 4, train_val_test_split = [0.7, 0.15, 0.15]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dsprites_train = None
        self.dsprites_val = None
        self.dsprites_test = None


    def prepare_data(self):
        DSpritesDataset(self.data_dir, download=True)

    def setup(self, stage=None):
        if not self.dsprites_train and not self.dsprites_val and not self.dsprites_test: # Esegui solo una volta
            dsprites_full = DSpritesDataset(self.data_dir, download=False, transform=self.transform)
            
            total_len = len(dsprites_full)
            train_len = int(self.train_val_test_split[0] * total_len)
            val_len = int(self.train_val_test_split[1] * total_len)
            test_len = total_len - train_len - val_len

            self.dsprites_train, self.dsprites_val, self.dsprites_test = random_split(
                dsprites_full, [train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(42) # Per riproducibilità dello split
            )
            print(f"Dataset split: Train={len(self.dsprites_train)}, Val={len(self.dsprites_val)}, Test={len(self.dsprites_test)}")


    def train_dataloader(self):
        return DataLoader(self.dsprites_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dsprites_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self): # Nuovo metodo
        return DataLoader(self.dsprites_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False) # Shuffle False per test riproducibili se si calcolano metriche