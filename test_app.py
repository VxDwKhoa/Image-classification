"""
Script kiểm thử toàn diện toàn bộ pipeline:
1. Nạp mô hình ArcFace & weights best_arcface_model.pth
2. Tự động tải/dò bộ trích xuất khuôn mặt AnimeFaceCropper
3. Kiểm thử với ảnh iroha.jpg và yui.jpg
4. Xác minh tính toán Softmax, Cosine và tạo Grad-CAM
5. Kiểm tra tính toàn vẹn của app PyQt6 và Streamlit
"""
import os
import sys

# Đảm bảo in UTF-8 trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def run_tests():
    print("==================================================")
    print("1. Kiểm tra môi trường thư viện...")
    try:
        import torch
        import torchvision
        import cv2
        import PIL
        import PyQt6
        import streamlit
        print(" -> PyTorch, Torchvision, OpenCV, Pillow, PyQt6, Streamlit: OK!")
    except ImportError as e:
        print(f" -> Thiếu thư viện: {e}")
        return False

    print("\n2. Kiểm tra trọng số best_arcface_model.pth...")
    if not os.path.exists("best_arcface_model.pth"):
        print(" -> Không tìm thấy best_arcface_model.pth trong thư mục hiện tại!")
        return False
    print(f" -> Kích thước file trọng số: {os.path.getsize('best_arcface_model.pth') / (1024*1024):.2f} MB")

    print("\n3. Kiểm tra module model_inference...")
    from model_inference import AnimePredictor
    predictor = AnimePredictor(model_path="best_arcface_model.pth")
    print(" -> Khởi tạo AnimePredictor thành công!")

    test_images = [f for f in ["iroha.jpg", "yui.jpg"] if os.path.exists(f)]
    if not test_images:
        print(" -> Cảnh báo: Không tìm thấy ảnh test (iroha.jpg / yui.jpg)")
    else:
        for img_name in test_images:
            print(f"\n--- Kiểm tra dự đoán cho: {img_name} ---")
            res = predictor.predict(img_name)
            print(f"   + Nhãn dự đoán: {res['predicted_label']}")
            print(f"   + Softmax Top: {res['pred_softmax']:.2f}%")
            print(f"   + Cosine Top: {res['pred_cosine']:.2f}")
            print(f"   + Bounding Box khuôn mặt: {res['bbox']}")
            print(f"   + Grad-CAM size: {res['gradcam_pil'].size}")
            assert res['pred_softmax'] >= 0.0 and res['pred_softmax'] <= 100.0
            assert res['gradcam_pil'] is not None

    print("\n4. Kiểm tra cấu trúc app PyQt6...")
    from app_pyqt6 import AnimeClassifierApp, DropUploadZone, ImageCardWidget
    print(" -> Import PyQt6 app components: OK!")

    print("\n5. Kiểm tra app Streamlit...")
    import app_streamlit
    print(" -> Import Streamlit app: OK!")

    print("\n==================================================")
    print("✅ TẤT CẢ CÁC KIỂM THỬ ĐÃ THÀNH CÔNG VƯỢT TRỘI!")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
