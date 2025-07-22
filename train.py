#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ── silence specific PIL transparency warning ─────────────────────────
import warnings
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
)

# ── stdlib / third-party ──────────────────────────────────────────────
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, accuracy_score

# Ultralytics YOLOv8 backbone
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # raised later if user picks yolov8*

# ───────────────────────── trainer class ──────────────────────────────
class ImageClassificationTrainer:
    def __init__(self, data_root: Path, arch: str, epochs: int,
                 batch_size: int, lr: float, imgsz: int,
                 output_dir: Path, device: torch.device):

        self.data_root   = data_root
        self.arch        = arch
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.imgsz       = imgsz
        self.output_dir  = output_dir
        self.device      = device

        # ── transforms (RGB enforced) ────────────────────────────────
        rgbify = transforms.Lambda(
            lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img
        )
        train_tf = transforms.Compose([
            rgbify,
            transforms.RandomResizedCrop(self.imgsz),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            rgbify,
            transforms.Resize(self.imgsz + 32),
            transforms.CenterCrop(self.imgsz),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        # ── datasets / loaders ───────────────────────────────────────
        self.train_loader = DataLoader(
            datasets.ImageFolder(self.data_root / 'train', transform=train_tf),
            batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )
        self.val_loader = DataLoader(
            datasets.ImageFolder(self.data_root / 'val', transform=val_tf),
            batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        self.test_loader = DataLoader(
            datasets.ImageFolder(self.data_root / 'test', transform=val_tf),
            batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        self.num_classes = len(self.train_loader.dataset.classes)

        # ── build model ──────────────────────────────────────────────
        self.model = self._build_backbone().to(self.device)

        # ── loss & optimiser ─────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # output dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------IMAGENET1K_V1----------------
    def _build_backbone(self) -> nn.Module:
        if self.arch.startswith("yolov8"):
            if YOLO is None:
                raise RuntimeError("Ultralytics not installed: pip install ultralytics")
            y = YOLO(f"{self.arch}.pt")          # downloads weights if absent
            backbone = y.model                   # raw nn.Module
        else:
            backbone = getattr(models, self.arch)(weights="IMAGENET1K_V1")

        # replace final fc / classifier
        for name, module in reversed(list(backbone.named_modules())):
            if isinstance(module, nn.Linear):
                parent = backbone
                *path, child = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                in_f = module.in_features
                setattr(parent, child, nn.Linear(in_f, self.num_classes))
                break
        else:
            raise RuntimeError("No nn.Linear layer found to replace.")
        return backbone

    # ───────────── training / evaluation helpers ─────────────────────
    def _forward(self, imgs):
        """Forward pass that returns logits tensor regardless of backbone."""
        out = self.model(imgs)
        if isinstance(out, (tuple, list)):   # YOLOv8 returns (logits,)
            out = out[0]
        return out

    def _run_epoch(self, loader, train: bool):
        self.model.train() if train else self.model.eval()
        loss_sum, preds_all, labels_all = 0.0, [], []

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.set_grad_enabled(train):
                logits = self._forward(imgs)
                loss   = self.criterion(logits, labels)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_sum += loss.item() * imgs.size(0)
            preds_all.append(logits.argmax(1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())

        avg_loss = loss_sum / len(loader.dataset)
        acc      = accuracy_score(np.concatenate(labels_all),
                                  np.concatenate(preds_all))
        return avg_loss, acc

    # ─────────────── main training loop ──────────────────────────────
    def fit(self):
        best_acc = 0.0
        best_ckp = self.output_dir / 'best_model.pth'

        for epoch in range(1, self.epochs + 1):
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            vl_loss, vl_acc = self._run_epoch(self.val_loader, train=False)
            print(f"Epoch {epoch:02d}/{self.epochs:02d} "
                  f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
                  f"Val {vl_loss:.4f}/{vl_acc:.4f}")

            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save(self.model.state_dict(), best_ckp)

        print(f"Best val accuracy: {best_acc:.4f}")
        return best_ckp

    # ───────────────────────── test & rename ─────────────────────────
    def test(self, checkpoint: Path):
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        ts_loss, ts_acc = self._run_epoch(self.test_loader, train=False)

        preds_all, labels_all = [], []
        for imgs, labels in self.test_loader:
            logits = self._forward(imgs.to(self.device))
            preds_all.append(logits.argmax(1).cpu().numpy())
            labels_all.append(labels.numpy())

        print(f"Test loss/acc: {ts_loss:.4f}/{ts_acc:.4f}")
        print(classification_report(
            np.concatenate(labels_all), np.concatenate(preds_all),
            target_names=self.test_loader.dataset.classes,
        ))

        # rename checkpoint with timestamp + test acc
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_ckpt  = checkpoint.with_name(f"{timestamp}_acc={ts_acc:.3f}.pth")
        checkpoint.rename(new_ckpt)
        print(f"Checkpoint saved as {new_ckpt.name}")

# ───────────────────────── CLI / entry-point ──────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Torch + YOLOv8 solar-panel classifier")
    p.add_argument('--data-root', type=Path, default=Path('split_solarpanel'))
    p.add_argument('--arch', default='yolov8n-cls',
                   help="e.g. yolov8n-cls, yolov8m-cls, resnet18, mobilenet_v3_large …")
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--imgsz', type=int, default=224)
    p.add_argument('--output-dir', type=Path, default=Path('runs/solarpanel_chkpt'))
    return p.parse_args()

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Torch version :", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    trainer = ImageClassificationTrainer(
        data_root=args.data_root,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
        device=device
    )

    best_path = trainer.fit()
    trainer.test(best_path)

if __name__ == '__main__':
    main()
