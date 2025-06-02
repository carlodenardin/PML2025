import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
        self.imgs = dataset_zip['imgs'] # These are (737280, 64, 64) numpy arrays, dtype=uint8
        self.latents_values = dataset_zip['latents_values']
        # latents_classes = dataset_zip['latents_classes']
        # metadata = dataset_zip['metadata'][()]

    def _download(self):
        print(f"Downloading dSprites dataset to {self.filepath}...")
        response = requests.get(DSPRITES_URL, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

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
        # Image is (64, 64), convert to (1, 64, 64) and float tensor
        image = self.imgs[idx].astype(np.float32) # Ensure float for ToTensor
        # ToTensor expects (H, W, C) or (H, W) if single channel.
        # Our images are (64,64). ToTensor will make it (1,64,64) and scale to [0,1]
        if self.transform:
            image = self.transform(image)
        return image

class DSpritesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/dsprites", batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor() # Converts numpy HWC/HW to CHW tensor and scales to [0,1]
        ])

    def prepare_data(self):
        # download only
        DSpritesDataset(self.data_dir, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dsprites_full = DSpritesDataset(self.data_dir, download=False, transform=self.transform)
            # Simple split, for a real project, consider fixed splits or more robust validation
            train_size = int(0.8 * len(dsprites_full))
            val_size = len(dsprites_full) - train_size
            self.dsprites_train, self.dsprites_val = torch.utils.data.random_split(
                dsprites_full, [train_size, val_size]
            )
        # Add test split if needed
        # if stage == 'test' or stage is None:
        #     self.dsprites_test = DSpritesDataset(self.data_dir, download=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.dsprites_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dsprites_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    # def test_dataloader(self):
    #     return DataLoader(self.dsprites_test, batch_size=self.batch_size, num_workers=self.num_workers)