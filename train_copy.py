import os
import argparse
import torch
from torch import nn, optim
from torch.optim import AdamW, SGD, RMSprop  # etc. if you need others
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings(
    'ignore',
    message='Palette images with Transparency expressed in bytes should be converted to RGBA images'
)


def split_dataset(dataset, train_ratio, val_ratio, test_ratio, seed):
    total = len(dataset)
    train_len = int(train_ratio * total)
    val_len = int(val_ratio * total)
    test_len = total - train_len - val_len
    lengths = [train_len, val_len, test_len]
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, lengths, generator=generator)

class BaseClassification:
    def __init__(self, data_dir, batch_size, epochs, lr, workers, seed,
                 train_ratio, val_ratio, test_ratio, imgsz=224):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.workers = workers
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.imgsz = imgsz

        torch.manual_seed(self.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose([
            transforms.Resize((self.imgsz, self.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(self.data_dir, transform=transform)
        self.classes = dataset.classes
        self.train_ds, self.val_ds, self.test_ds = split_dataset(
            dataset, self.train_ratio, self.val_ratio, self.test_ratio, self.seed)

        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size,
                                       shuffle=True, num_workers=self.workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size,
                                     shuffle=False, num_workers=self.workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.batch_size,
                                      shuffle=False, num_workers=self.workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)

class YOLOClassification(BaseClassification):
    def __init__(self, *args, arch='yolov8n-cls', **kwargs):
        super().__init__(*args, **kwargs)
        self.arch = arch
        self.model = self._build_backbone()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-2
        )

    def _safe_load_yolo(self):
        weight = f"{self.arch}.pt"
        if os.path.isfile(weight):
            return YOLO(weight)
        try:
            return YOLO(weight)
        except Exception:
            cfg = f"{self.arch}.yaml"
            return YOLO(cfg) if os.path.isfile(cfg) else YOLO(model=None, task='classify')

    def _build_backbone(self):
        """
        Load a YOLO-v8 backbone and **insert a small projection head**
        (BN → ReLU → Dropout → Linear) in front of the final classifier.
        """
        if not self.arch.startswith('yolov8'):
            raise RuntimeError('arch must start with yolov8')

        yolo = self._safe_load_yolo()
        backbone = yolo.model                                         # full nn.Module

        # ── locate the last Linear (the original classifier) ───────────────
        for name, module in reversed(list(backbone.named_modules())):
            if isinstance(module, nn.Linear):
                parent = backbone
                *path, child_name = name.split('.')
                for p in path:                                         # walk down to the parent
                    parent = getattr(parent, p)
                in_features = module.in_features

                # ── NEW projection head ────────────────────────────────────
                hidden = max(128, in_features // 2)                     # e.g. 512 for yolov8-n
                new_head = nn.Sequential(
                    nn.BatchNorm1d(in_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                    nn.Linear(in_features, hidden, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, len(self.classes), bias=True),
                )

                setattr(parent, child_name, new_head)                  # replace in-place
                break

        return backbone.to(self.device)

    def _forward(self, imgs):
        out = self.model(imgs)
        return out[0] if isinstance(out, (tuple, list)) else out

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss, preds, labels = 0.0, [], []
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)
            with torch.set_grad_enabled(train):
                logits = self._forward(imgs)
                loss = self.criterion(logits, lbls)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(lbls.cpu().numpy())
        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(np.concatenate(labels), np.concatenate(preds))
        return avg_loss, acc

    def fit(self, patience=5, min_delta=0.0):
        best_acc, best_ckp = 0.0, Path('best_yolo.pth')
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            tr_loss, tr_acc = self._run_epoch(self.train_loader, True)
            vl_loss, vl_acc = self._run_epoch(self.val_loader, False)
            print(f"Epoch {epoch}/{self.epochs} Train {tr_loss:.4f}/{tr_acc:.4f} | Val {vl_loss:.4f}/{vl_acc:.4f}")

            if vl_loss + min_delta < best_val_loss:
                best_val_loss = vl_loss
                epochs_no_improve = 0
                best_acc = vl_acc
                torch.save(self.model.state_dict(), best_ckp)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Best val acc: {best_acc:.4f}")
        return best_ckp

    def test(self, checkpoint):
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        ts_loss, ts_acc = self._run_epoch(self.test_loader, False)
        print(f"Test loss/acc: {ts_loss:.4f}/{ts_acc:.4f}")
        all_preds, all_labels = [], []
        for imgs, lbls in self.test_loader:
            logits = self._forward(imgs.to(self.device))
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(lbls.numpy())
        print(classification_report(np.concatenate(all_labels), np.concatenate(all_preds), target_names=self.classes))
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_ckp = Path(checkpoint).with_name(f"{ts}_acc={ts_acc:.3f}.pth")
        Path(checkpoint).rename(new_ckp)
        print(f"Saved checkpoint: {new_ckp.name}")

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='YOLOv8 solar panel classifier')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model', default='yolov8n-cls', help='YOLOv8 model name')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {'GPU' if device.type=='cuda' else 'CPU'} ({device})")
    clf = YOLOClassification(
        args.data_dir, args.batch_size, args.epochs,
        args.lr, args.workers, args.seed,
        args.train_ratio, args.val_ratio, args.test_ratio,
        args.imgsz, arch=args.model
    )
    ckp = clf.fit()
    clf.test(ckp)
