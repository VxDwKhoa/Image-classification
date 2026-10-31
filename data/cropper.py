import os
import urllib.request
from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image


class AnimeFaceCropper:
    """
    Bộ phát hiện và cắt vùng khuôn mặt nhân vật anime sử dụng LBP/Haar Cascade.
    Hỗ trợ cơ chế dự phòng Top-Center Crop 70% khi không tìm thấy mặt.
    """

    def __init__(
        self,
        cascade_path: str = "lbpcascade_animeface.xml",
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.target_size = target_size
        self.cascade_path = cascade_path

        # Tự động tải file cascade nếu chưa tồn tại
        if not os.path.exists(cascade_path):
            cascade_url = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
            try:
                urllib.request.urlretrieve(cascade_url, cascade_path)
            except Exception as e:
                print(f"[AnimeFaceCropper] Không thể tải XML cascade: {e}")

        self.face_cascade = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else None

    def crop(self, pil_img: Image.Image) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
        """
        Nhận ảnh PIL RGB và trả về:
        - cropped_img: Ảnh PIL cận cảnh khuôn mặt (target_size)
        - bbox: Bounding box trên ảnh gốc (x1, y1, x2, y2)
        """
        img_np = np.array(pil_img)
        h_img, w_img = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        faces = []
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=7, minSize=(50, 50)
            )

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            # Thêm viền đệm 20% để lấy trọn vẹn tóc và phụ kiện
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)

            cropped = img_np[y1:y2, x1:x2]
            bbox = (x1, y1, x2, y2)
        else:
            # Fallback: Cắt Top-Center (70% diện tích lệch lên phía trên)
            crop_size = int(min(h_img, w_img) * 0.7)
            start_y = int(h_img * 0.1)
            start_x = (w_img - crop_size) // 2

            if start_y + crop_size > h_img:
                start_y = h_img - crop_size

            cropped = img_np[start_y : start_y + crop_size, start_x : start_x + crop_size]
            bbox = (start_x, start_y, start_x + crop_size, start_y + crop_size)

        cropped_resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
        return Image.fromarray(cropped_resized), bbox
