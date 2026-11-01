import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace).
    Tối ưu hóa góc phân tách trên siêu cầu đơn vị:
        cos(theta + m) với m = margin góc, s = scale phóng đại bán kính.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.30,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Chuẩn hóa L2 cho feature embedding và ma trận trọng số
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # Tính cosine góc theta giữa embedding và trọng số lớp
	#X norm,Y norm sau khi chuản hóa có độ dài là 1 nên cosin bằng đúng tính vô hướng của hai ma trận 
        cos_theta = F.linear(x_norm, w_norm)
        cos_theta = torch.clamp(cos_theta, min=-1.0 + 1e-7, max=1.0 - 1e-7)

        # Khi ở chế độ Test/Inference (labels is None): không cộng góc phạt
	# dùng cho  test
        if labels is None:
            return self.s * cos_theta, cos_theta

        # Khi ở chế độ Train (labels provided): áp dụng góc phạt margin m
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=1e-7))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        one_hot = F.one_hot(labels, num_classes=self.out_features).to(cos_theta.dtype)
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        logits = self.s * output

        return logits, cos_theta
