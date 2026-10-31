import os
from typing import List
import pandas as pd
from PIL import Image
import streamlit as st

from model_inference import AnimePredictor

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="Nhận diện nhân vật anime & Grad-CAM",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS cho phong cách hiện đại và giống hệt ảnh mẫu
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .img-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .col-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.4rem;
    }
    .prediction-box {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .stTable {
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return AnimePredictor(model_path="best_arcface_model.pth")


def main():
    st.markdown('<div class="main-title">🔍 Nhận diện nhân vật anime & Bản đồ nhiệt Grad-CAM</div>', unsafe_allow_html=True)

    # 1. Bộ nạp ảnh
    uploaded_files = st.file_uploader(
        "Kéo thả một hoặc nhiều bức ảnh lên đây...",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"📁 Đã tải lên {len(uploaded_files)} bức ảnh. Hệ thống đang phân tích...")

        with st.spinner("Đang tải mô hình ArcFace & Grad-CAM..."):
            try:
                predictor = load_predictor()
            except Exception as e:
                st.error(f"Lỗi nạp mô hình: {e}")
                return

        # Hiển thị từng ảnh theo hàng tuần tự
        for idx, file in enumerate(uploaded_files):
            file_bytes = file.read()
            from io import BytesIO
            pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")

            # Chạy inference
            with st.spinner(f"Đang phân tích ảnh {idx + 1}: {file.name}..."):
                res = predictor.predict(pil_img, file_name=file.name)

            # Tiêu đề ảnh
            st.markdown(f'<div class="img-title">📷 Ảnh {idx + 1}: <code style="color: #059669;">{file.name}</code></div>', unsafe_allow_html=True)

            # Khung kết quả dự đoán
            st.markdown(
                f"""
                <div class="prediction-box">
                    ✅ <b>Dự đoán:</b> <b>{res['predicted_label']}</b> 
                    (Softmax: <b>{res['pred_softmax']:.2f}%</b> | Cosine: <b>{res['pred_cosine']:.2f}</b>)
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Layout 4 cột đúng mẫu
            col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.1, 1.4])

            with col1:
                st.markdown('<div class="col-header">1. Ảnh Toàn cục</div>', unsafe_allow_html=True)
                st.image(res["global_pil"], use_container_width=True)

            with col2:
                st.markdown('<div class="col-header">2. Local</div>', unsafe_allow_html=True)
                st.image(res["local_pil"], use_container_width=True)

            with col3:
                st.markdown('<div class="col-header">3. Tâm điểm (Grad-CAM)</div>', unsafe_allow_html=True)
                st.image(res["gradcam_pil"], use_container_width=True)

            with col4:
                st.markdown('<div class="col-header">4. Xác suất</div>', unsafe_allow_html=True)
                df = pd.DataFrame([
                    {
                        "Nhân vật": r["character"],
                        "Xác suất Softmax (%)": f"{r['softmax']:.2f}%" if r["softmax"] >= 0.01 else "0",
                        "Cosine Gốc": f"{r['cosine']:.2f}",
                    }
                    for r in res["table_rows"]
                ])
                st.dataframe(df, hide_index=True, use_container_width=True)

            st.divider()


if __name__ == "__main__":
    main()
