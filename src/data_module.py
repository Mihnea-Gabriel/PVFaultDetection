import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional
from PIL import Image
import warnings

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
    category=UserWarning,
    module="PIL.Image",
)

warnings.filterwarnings(
    "ignore",
    message="The parameter 'pretrained' is deprecated since 0.13 ",
    category=UserWarning,
    module="torchvision",
)

warnings.filterwarnings(
    "ignore",
    message="Arguments other than a weight enum or `None` for 'weights' ",
    category=UserWarning,
    module="torchvision",
)

class DataModule :
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        img_size: Tuple[int, int],
        train_split: float,
        val_split: float,
        test_split : float,
        num_workers: int,
        seed: int,
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.img_size = img_size
        self.val_split = val_split
        self.train_split = train_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.seed = seed

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406],
                                 std=[.229, .224, .225]),
        ])

    def setup(self) -> None:
        full_ds = datasets.ImageFolder(root = self.data_dir, transform = self.transform)
        total_len = len(full_ds)

        test_len = int(total_len * self.test_split)
        train_len = int(total_len * self.train_split)
        val_len = total_len - test_len - train_len

        train_ds, val_ds, test_ds = random_split(
                                                full_ds,
                                                [train_len, val_len, test_len],
                                                generator=torch.Generator().manual_seed(self.seed)
                                                )


        self.full_ds = full_ds
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.classes = full_ds.classes


    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
                                 self.train_ds,
                                 batch_size = self.batch_size,
                                 shuffle = True,
                                 num_workers = self.num_workers   
        )
        val_loader = DataLoader(
                                 self.val_ds,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 num_workers = self.num_workers   
        )
        test_loader = DataLoader(
                                 self.test_ds,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 num_workers = self.num_workers   
        )
        fullds_loader = DataLoader(
                                 self.full_ds,
                                 batch_size = self.batch_size,
                                 shuffle = True,
                                 num_workers = self.num_workers   
        )

        return train_loader, val_loader, test_loader, fullds_loader

