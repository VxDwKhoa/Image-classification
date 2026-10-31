import argparse
import os
import torch
from torch.utils.data import DataLoader

from models import MyModel
from data import (
    OregairuDualDataset,
    train_global_transform,
    train_local_transform,
    eval_transform,
)
from training import ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Dual EfficientNet-B0 + ArcFace")
    parser.add_argument("--train_dir", type=str, required=True, help="Đường dẫn thư mục tập train")
    parser.add_argument("--val_dir", type=str, required=True, help="Đường dẫn thư mục tập val/test")
    parser.add_argument("--epochs", type=int, default=15, help="Số epochs huấn luyện (mặc định: 15)")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch (mặc định: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (mặc định: 1e-4)")
    parser.add_argument("--num_classes", type=int, default=3, help="Số lớp nhân vật (mặc định: 3)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Thư mục lưu weights")
    parser.add_argument("--cache_dir", type=str, default=None, help="Thư mục cache ảnh khuôn mặt")
    return parser.parse_args()


def main():
    args = parse_args()

    print("==========================================================")
    print("   KHỞI ĐỘNG PIPELINE HUẤN LUYỆN ANIME DUAL EFFICIENTNET  ")
    print("==========================================================")

    # 1. Dataset & DataLoader
    print(f"📦 Đang nạp tập Train từ: {args.train_dir}")
    train_dataset = OregairuDualDataset(
        root_dir=args.train_dir,
        transform_global=train_global_transform,
        transform_local=train_local_transform,
        cache_dir=args.cache_dir,
    )

    print(f"📦 Đang nạp tập Val từ: {args.val_dir}")
    val_dataset = OregairuDualDataset(
        root_dir=args.val_dir,
        transform_global=eval_transform,
        transform_local=eval_transform,
        cache_dir=args.cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"-> Tổng mẫu Train: {len(train_dataset)} | Tổng mẫu Val: {len(val_dataset)}")
    print(f"-> Danh sách các lớp: {train_dataset.classes}")

    # 2. Khởi tạo Mô hình & Trọng số Pretrained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Khởi tạo MyModel (ArcFace) trên thiết bị: {device}")
    model = MyModel(num_classes=args.num_classes, pretrained=True).to(device)

    # 3. Khởi tạo Optimizer & Trainer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # 4. Chạy huấn luyện
    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
