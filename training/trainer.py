import os
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ModelTrainer:
    """
    Quản lý vòng lặp huấn luyện, đánh giá (Validation), và lưu Checkpoint tốt nhất.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Huấn luyện 1 epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for img_global, img_local, labels in self.train_loader:
            img_global = img_global.to(self.device, non_blocking=True)
            img_local = img_local.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            # Bắt buộc truyền labels để kích hoạt góc phạt m của ArcFace trong lúc train
            logits, _ = self.model(img_global, img_local, labels)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * img_global.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += img_global.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        return epoch_loss, epoch_acc

    def evaluate(self) -> Tuple[float, float]:
        """Đánh giá 1 epoch trên tập validation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for img_global, img_local, labels in self.val_loader:
                img_global = img_global.to(self.device, non_blocking=True)
                img_local = img_local.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Khi Test/Eval: Không truyền labels (labels=None)
                logits, _ = self.model(img_global, img_local)
                loss = self.criterion(logits, labels)

                running_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total
        val_acc = (correct / total) * 100.0
        return val_loss, val_acc

    def fit(self, epochs: int = 15, save_name: str = "best_arcface_model.pth") -> Dict[str, list]:
        """Vòng lặp huấn luyện chính qua các epoch."""
        best_val_acc = 0.0
        best_model_path = os.path.join(self.checkpoint_dir, save_name)

        print(f"🚀 Bắt đầu huấn luyện {epochs} epochs trên thiết bị: {self.device}")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            saved_flag = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), best_model_path)
                saved_flag = " ⭐ [Đã lưu Best Checkpoint]"

            print(
                f"Epoch [{epoch:2d}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} - Acc: {train_acc:6.2f}% | "
                f"Val Loss: {val_loss:.4f} - Acc: {val_acc:6.2f}%{saved_flag}"
            )

        print("-" * 75)
        print(f"✅ Hoàn thành! Best Val Acc: {best_val_acc:.2f}%")
        print(f"📦 Checkpoint tốt nhất đã được lưu tại: {best_model_path}")
        return self.history
