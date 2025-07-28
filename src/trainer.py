import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional

class Trainer: 
    def __init__(self,
                 model: nn.Module,
                 model_name : str,
                 train_loader,
                 val_loader,
                 test_loader,
                 device: torch.device,
                 lr: float,
                 patience:  int
                 ):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.best_val_acc: Optional[float] = None
        self.runs_dir = Path("/home/expleo2/Desktop/MyProject/runs") 
        self.runs_dir.mkdir(parents=True, exist_ok=True) 
        self.patience = patience


    def run_epoch(self, loader, train: bool):
        target = self.model.model if hasattr(self.model, "model") else self.model
        torch.nn.Module.train(target, train)     

        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            logits = self.model(imgs)
            loss = self.criterion(logits, labels)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / len(loader.dataset), correct / total
        

    def fit(self, epochs: int) :
            epochs_no_improve = 0 
            for epoch in range(1, epochs + 1) : 
                train_loss, train_acc = self.run_epoch(self.train_loader, train=True)
                val_loss, val_acc = self.run_epoch(self.val_loader, train=False)

                print(
                    f"[Epoch {epoch:02d}] "
                    f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
                    f"val_loss:   {val_loss:.4f} val_acc:   {val_acc:.4f}"
                )

                improved = (self.best_val_acc is None or val_acc > self.best_val_acc)

                if improved:
                        self.best_val_acc = val_acc
                        fname = f"{self.model_name}_valacc{val_acc:.4f}.pth"
                        checkpoint_path = self.runs_dir/fname
                        epochs_no_improve = 0
                        torch.save(
                            {
                                "model_state_dict": self.model.state_dict(),
                                "model_name": self.model_name,       
                                "class_names": self.model.class_names,
                            },
                            checkpoint_path,
                                )
                        print(f"  → New best; saved to {checkpoint_path}")
                else :
                     epochs_no_improve +=1

                if epochs_no_improve >= self.patience :
                     print(f"Early Stopping")
                     break



            test_loss, test_acc = self.run_epoch(self.test_loader, train=False)
            print(f"[TEST] loss: {test_loss:.4f}  acc: {test_acc:.4f}")
