from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from .arcface import ArcMarginProduct


class MyModel(nn.Module):
    """
    Kiến trúc Dual-Branch EfficientNet-B0 kết hợp ArcFace Head.
    - Nhánh 1 (Global): Trích xuất đặc trưng toàn thể ảnh (tóc, dáng, trang phục).
    - Nhánh 2 (Local): Trích xuất đặc trưng local
    - Feature Fusion: Nối 2 vector 1280-chiều -> 2560-chiều.
    - Neck: BatchNorm1d -> Dropout(0.5) -> Linear -> Vector embedding 512-chiều.
    - Head: ArcMarginProduct.
    """

    def __init__(
        self,
        num_classes: int = 3,
        embedding_dim: int = 512,
        s: float = 30.0,
        m: float = 0.30,
        pretrained: bool = False,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_b0(weights=weights)
        self.backbone = base_model.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Neck Block kết hợp
        self.neck = nn.Sequential(
            nn.BatchNorm1d(num_features=1280 * 2),
            nn.Dropout(p=0.5),
            nn.Linear(1280 * 2, embedding_dim),
        )

        # ArcFace Head
        self.arcface = ArcMarginProduct(
            in_features=embedding_dim, out_features=num_classes, s=s, m=m
        )

    def extract_branch_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất feature vector từ 1 nhánh ảnh."""
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x
    # Hàm lan truyền thuận
    def forward(
        self,
        img_global: torch.Tensor,
        img_local: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_global = self.extract_branch_feature(img_global)
        feat_local = self.extract_branch_feature(img_local)

        # Hợp nhất đặc trưng hai nhánh
        merged = torch.cat([feat_global, feat_local], dim=1)
        embeddings = self.neck(merged)

        logits, cos_theta = self.arcface(embeddings, labels)
        return logits, cos_theta
