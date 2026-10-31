import os
from typing import Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from .cropper import AnimeFaceCropper

# Chuẩn hóa ImageNet
imagenet_norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# 1. Pipeline Augmentation cho tập TRAIN
train_global_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    imagenet_norm,
])

train_local_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    imagenet_norm,
])

# 2. Pipeline biến đổi cho tập EVAL / TEST (Giữ nguyên kích thước 224x224 và chuẩn hóa)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    imagenet_norm,
])


class OregairuDualDataset(Dataset):
    """
    Dataset tải ảnh theo 2 nhánh:
    - img_global: Ảnh toàn cảnh (Global Context)
    - img_local: Ảnh khuôn mặt sau khi cắt (Local Face ROI)
    Hỗ trợ cơ chế cache ảnh khuôn mặt vào đĩa để tránh nghẽn CPU trong lúc huấn luyện.
    """

    def __init__(
        self,
        root_dir: str,
        transform_global: Optional[Callable] = None,
        transform_local: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
    ):
        self.root_dir = root_dir
        self.transform_global = transform_global
        self.transform_local = transform_local
        self.cache_dir = cache_dir

        self.cropper = AnimeFaceCropper(target_size=(224, 224))
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.image_paths = []
        self.labels = []

        # 1. Quét danh sách lớp
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Thư mục dataset '{root_dir}' không tồn tại.")

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 2. Duyệt qua từng thư mục gom đường dẫn ảnh và nhãn
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            label_idx = self.class_to_idx[cls_name]

            for img_name in os.listdir(cls_folder):
                ext = os.path.splitext(img_name)[1].lower()
                if ext in valid_exts:
                    img_path = os.path.join(cls_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label_idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 1. Đọc ảnh gốc (cho nhánh Global)
        img_raw = Image.open(img_path).convert("RGB")

        # 2. Xử lý ảnh khuôn mặt (Local) với bộ nhớ đệm Cache
        img_local_raw = None
        if self.cache_dir is not None:
            img_filename = os.path.basename(img_path)
            parent_folder = os.path.basename(os.path.dirname(img_path))
            cache_file = os.path.join(self.cache_dir, f"{parent_folder}_{img_filename}")

            if os.path.exists(cache_file):
                img_local_raw = Image.open(cache_file).convert("RGB")
            else:
                img_local_raw, _ = self.cropper.crop(img_raw)
                img_local_raw.save(cache_file)
        else:
            img_local_raw, _ = self.cropper.crop(img_raw)

        # 3. Áp dụng transforms riêng cho từng nhánh
        img_global = img_raw
        img_local = img_local_raw

        if self.transform_global is not None:
            img_global = self.transform_global(img_global)
        if self.transform_local is not None:
            img_local = self.transform_local(img_local)

        return img_global, img_local, label
