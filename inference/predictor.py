import os
from typing import Dict, Any, List, Optional, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.dual_efficientnet import MyModel
from data.cropper import AnimeFaceCropper
from data.dataset import eval_transform


class AnimePredictor:
    """
    Engine suy luận dự đoán nhân vật anime kết hợp trực quan hóa Grad-CAM.
    """

    CLASS_NAMES = [
        "Iroha Isshiki",
        "Yui Yuigahama",
        "Yukino Yukinoshita",
    ]

    def __init__(
        self,
        model_path: str = "best_arcface_model.pth",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = MyModel(num_classes=3).to(self.device)
        self._load_weights(model_path)
        self.model.eval()

        self.cropper = AnimeFaceCropper(target_size=(224, 224))
        self.eval_transform = eval_transform

    def _load_weights(self, model_path: str):
        candidates = [
            model_path,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", model_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path),
            r"D:\Tai Xuong\ML\best_arcface_model.pth",
            r"D:\C++\ML\best_arcface_model.pth",
        ]
        found_path = None
        for path in candidates:
            if os.path.exists(path):
                found_path = path
                break

        if found_path is None:
            raise FileNotFoundError(
                f"Khong tim thay file trong so '{model_path}'. Vui long kiem tra duong dan."
            )

        state_dict = torch.load(found_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.loaded_path = found_path
        print(f"Loaded weights from: {found_path} on device {self.device}")

    def predict(
        self,
        image_input: Union[str, Image.Image],
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dự đoán cho một ảnh và tính toán Grad-CAM:
        - image_input: Đường dẫn str hoặc đối tượng PIL.Image
        """
        if isinstance(image_input, str):
            if file_name is None:
                file_name = os.path.basename(image_input)
            img_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img_pil = image_input.convert("RGB")
            if file_name is None:
                file_name = "image_input.jpg"
        else:
            raise TypeError("image_input phải là đường dẫn str hoặc PIL.Image")

        # 1. Cắt khuôn mặt (Local)
        cropped_local_pil, bbox = self.cropper.crop(img_pil)

        # 2. Chuẩn bị tensor đầu vào
        t_global = self.eval_transform(img_pil).unsqueeze(0).to(self.device)
        t_local = self.eval_transform(cropped_local_pil).unsqueeze(0).to(self.device)

        # 3. Forward pass và kích hoạt retain_grad cho nhánh Local để tính Grad-CAM
        f_global = self.model.backbone(t_global)
        feat_global = self.model.flatten(self.model.pool(f_global))

        f_local = self.model.backbone(t_local)
        f_local.retain_grad()
        feat_local = self.model.flatten(self.model.pool(f_local))

        merged = torch.cat([feat_global, feat_local], dim=1)
        embeddings = self.model.neck(merged)
        logits, cos_theta = self.model.arcface(embeddings)

        pred_idx = torch.argmax(logits, dim=1).item()
        score = logits[0, pred_idx]

        # Lan truyền ngược lấy gradient tính Grad-CAM
        self.model.zero_grad()
        score.backward()

        grads = f_local.grad
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * f_local).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam_np.min(), cam_np.max()
        cam_norm = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        cropped_cv = cv2.resize(np.array(cropped_local_pil), (224, 224))
        cam_overlay = np.uint8(0.45 * heatmap + 0.55 * cropped_cv)
        cam_overlay_pil = Image.fromarray(cam_overlay)

        # 4. Xác suất Softmax và Cosine
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy() * 100.0
        cosines = cos_theta[0].detach().cpu().numpy()

        predicted_label = self.CLASS_NAMES[pred_idx]
        pred_softmax = float(probs[pred_idx])
        pred_cosine = float(cosines[pred_idx])

        table_rows = []
        for i, class_name in enumerate(self.CLASS_NAMES):
            table_rows.append({
                "character": class_name,
                "softmax": float(probs[i]),
                "cosine": float(cosines[i]),
                "is_winner": (i == pred_idx),
            })

        return {
            "file_name": file_name,
            "global_pil": img_pil,
            "local_pil": cropped_local_pil,
            "gradcam_pil": cam_overlay_pil,
            "pred_idx": pred_idx,
            "predicted_label": predicted_label,
            "pred_softmax": pred_softmax,
            "pred_cosine": pred_cosine,
            "table_rows": table_rows,
            "bbox": bbox,
        }
