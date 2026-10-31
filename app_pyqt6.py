import os
import sys
from typing import List, Dict, Any, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QPalette, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QSizePolicy,
)
from PIL import Image

from model_inference import AnimePredictor


def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    """Chuyển đổi PIL Image sang QPixmap để hiển thị trên QLabel."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    data = pil_img.tobytes("raw", "RGB")
    qimg = QImage(data, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class InferenceWorker(QThread):
    """Thread chạy ngầm để inference không làm đơ giao diện khi xử lý nhiều ảnh."""
    progress = pyqtSignal(int, int)
    image_processed = pyqtSignal(dict)
    finished_all = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, predictor: AnimePredictor, file_paths: List[str]):
        super().__init__()
        self.predictor = predictor
        self.file_paths = file_paths

    def run(self):
        total = len(self.file_paths)
        for idx, path in enumerate(self.file_paths):
            try:
                result = self.predictor.predict(path)
                result["index"] = idx + 1
                self.image_processed.emit(result)
            except Exception as e:
                self.error_occurred.emit(f"Lỗi xử lý file {os.path.basename(path)}: {str(e)}")
            self.progress.emit(idx + 1, total)
        self.finished_all.emit()


class DropUploadZone(QFrame):
    """Vùng kéo thả ảnh trực quan tương tự UI web."""
    files_dropped = pyqtSignal(list)
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            DropUploadZone {
                background-color: #f8fafc;
                border: 2px dashed #94a3b8;
                border-radius: 12px;
                padding: 24px;
            }
            DropUploadZone:hover {
                background-color: #f1f5f9;
                border-color: #3b82f6;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        icon_label = QLabel("📥", self)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 38px; border: none; background: transparent;")

        title_label = QLabel("Kéo thả một hoặc nhiều bức ảnh lên đây...", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #334155; border: none; background: transparent;")

        sub_label = QLabel("Hỗ trợ JPG, JPEG, PNG, WEBP (Nhấp chuột hoặc nhấn nút bên dưới để chọn)", self)
        sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub_label.setStyleSheet("font-size: 12px; color: #64748b; border: none; background: transparent;")

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(sub_label)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                DropUploadZone {
                    background-color: #eff6ff;
                    border: 2px dashed #2563eb;
                    border-radius: 12px;
                    padding: 24px;
                }
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            DropUploadZone {
                background-color: #f8fafc;
                border: 2px dashed #94a3b8;
                border-radius: 12px;
                padding: 24px;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        valid_files = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for url in urls:
            fpath = url.toLocalFile()
            if os.path.isfile(fpath) and os.path.splitext(fpath)[1].lower() in valid_exts:
                valid_files.append(fpath)

        self.setStyleSheet("""
            DropUploadZone {
                background-color: #f8fafc;
                border: 2px dashed #94a3b8;
                border-radius: 12px;
                padding: 24px;
            }
        """)

        if valid_files:
            self.files_dropped.emit(valid_files)


class ImageCardWidget(QFrame):
    """Khung thẻ hiển thị kết quả cho từng bức ảnh theo chuẩn 4 cột."""
    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            ImageCardWidget {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                margin-bottom: 20px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        # 1. Header ảnh: 📷 Ảnh {idx}: {filename}
        header_text = f"📷 Ảnh {data.get('index', 1)}:  {data['file_name']}"
        header_label = QLabel(header_text, self)
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e293b;")
        main_layout.addWidget(header_label)

        # 2. Banner dự đoán màu xanh lá nhẹ: ✅ Dự đoán: {Tên} (Softmax: ...% | Cosine: ...)
        banner_text = (
            f"✅ <b>Dự đoán:</b> <span style='font-weight: 600;'>{data['predicted_label']}</span> "
            f"(Softmax: <b>{data['pred_softmax']:.2f}%</b> | Cosine: <b>{data['pred_cosine']:.2f}</b>)"
        )
        banner_label = QLabel(banner_text, self)
        banner_label.setTextFormat(Qt.TextFormat.RichText)
        banner_label.setStyleSheet("""
            QLabel {
                background-color: #ecfdf5;
                color: #065f46;
                border: 1px solid #a7f3d0;
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 14px;
            }
        """)
        main_layout.addWidget(banner_label)

        # 3. Layout 4 cột theo đúng mẫu
        cols_layout = QHBoxLayout()
        cols_layout.setSpacing(16)

        # --- Cột 1: Ảnh Toàn cục ---
        col1 = self._build_image_column("1. Ảnh Toàn cục", data["global_pil"])
        cols_layout.addWidget(col1, stretch=1)

        # --- Cột 2: Local ---
        col2 = self._build_image_column("2. Local", data["local_pil"])
        cols_layout.addWidget(col2, stretch=1)

        # --- Cột 3: Tâm điểm (Grad-CAM) ---
        col3 = self._build_image_column("3. Tâm điểm (Grad-CAM)", data["gradcam_pil"])
        cols_layout.addWidget(col3, stretch=1)

        # --- Cột 4: Bảng Xác suất ---
        col4 = self._build_table_column("4. Xác suất", data["table_rows"])
        cols_layout.addWidget(col4, stretch=1)

        main_layout.addLayout(cols_layout)

    def _build_image_column(self, title: str, pil_img: Image.Image) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title_lbl = QLabel(title, container)
        title_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #334155;")
        layout.addWidget(title_lbl)

        img_lbl = QLabel(container)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setStyleSheet("""
            QLabel {
                background-color: #f1f5f9;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
            }
        """)
        img_lbl.setFixedSize(224, 224)

        pixmap = pil_to_qpixmap(pil_img)
        scaled_pixmap = pixmap.scaled(
            224, 224,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        img_lbl.setPixmap(scaled_pixmap)
        layout.addWidget(img_lbl)
        layout.addStretch()

        return container

    def _build_table_column(self, title: str, table_rows: List[dict]) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title_lbl = QLabel(title, container)
        title_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #334155;")
        layout.addWidget(title_lbl)

        table = QTableWidget(len(table_rows), 3, container)
        table.setHorizontalHeaderLabels(["Nhân vật", "Xác suất Softmax (%)", "Cosine Gốc"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: #f1f5f9;
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #f8fafc;
                color: #475569;
                font-weight: bold;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #cbd5e1;
            }
        """)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        for row, row_data in enumerate(table_rows):
            is_winner = row_data["is_winner"]

            # Nhân vật
            item_char = QTableWidgetItem(row_data["character"])
            # Softmax
            item_soft = QTableWidgetItem(f"{row_data['softmax']:.2f}%")
            item_soft.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # Cosine
            item_cos = QTableWidgetItem(f"{row_data['cosine']:.2f}")
            item_cos.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            if is_winner:
                font = QFont()
                font.setBold(True)
                item_char.setFont(font)
                item_soft.setFont(font)
                item_cos.setFont(font)

                win_bg = QColor("#ecfdf5")
                item_char.setBackground(win_bg)
                item_soft.setBackground(win_bg)
                item_cos.setBackground(win_bg)

                win_fg = QColor("#065f46")
                item_char.setForeground(win_fg)
                item_soft.setForeground(win_fg)
                item_cos.setForeground(win_fg)

            table.setItem(row, 0, item_char)
            table.setItem(row, 1, item_soft)
            table.setItem(row, 2, item_cos)

        table.setFixedHeight(224)
        layout.addWidget(table)
        layout.addStretch()

        return container


class AnimeClassifierApp(QMainWindow):
    """Cửa sổ ứng dụng chính PyQt6."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nhận diện nhân vật anime & Bản đồ nhiệt Grad-CAM")
        self.resize(1150, 850)
        self.setMinimumSize(950, 700)

        # Tải mô hình thông qua AnimePredictor
        self.predictor = None
        self._init_predictor()

        self._init_ui()

    def _init_predictor(self):
        try:
            self.predictor = AnimePredictor()
        except Exception as e:
            print(f"Lỗi khởi tạo mô hình: {e}")

    def _init_ui(self):
        # Thiết lập style tổng thể
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8fafc;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            QPushButton#uploadBtn {
                background-color: #2563eb;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
                border: none;
            }
            QPushButton#uploadBtn:hover {
                background-color: #1d4ed8;
            }
            QPushButton#clearBtn {
                background-color: #f1f5f9;
                color: #475569;
                font-size: 13px;
                border-radius: 8px;
                padding: 8px 16px;
                border: 1px solid #cbd5e1;
            }
            QPushButton#clearBtn:hover {
                background-color: #e2e8f0;
            }
        """)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(16)

        # 1. Header Title
        title_box = QHBoxLayout()
        title_label = QLabel("🔍 Nhận diện nhân vật anime & Bản đồ nhiệt Grad-CAM", self)
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #0f172a;")
        title_box.addWidget(title_label)
        title_box.addStretch()

        self.clear_btn = QPushButton("🗑️ Xóa danh sách", self)
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self.clear_results)
        self.clear_btn.setVisible(False)
        title_box.addWidget(self.clear_btn)

        main_layout.addLayout(title_box)

        # 2. Vùng kéo thả file
        self.drop_zone = DropUploadZone(self)
        self.drop_zone.files_dropped.connect(self.handle_files)
        self.drop_zone.clicked.connect(self.browse_files)
        main_layout.addWidget(self.drop_zone)

        # Nút chọn file thủ công
        btn_layout = QHBoxLayout()
        self.browse_btn = QPushButton("📁 Chọn ảnh từ máy tính (Có thể chọn nhiều ảnh)...", self)
        self.browse_btn.setObjectName("uploadBtn")
        self.browse_btn.clicked.connect(self.browse_files)
        btn_layout.addStretch()
        btn_layout.addWidget(self.browse_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # 3. Status Bar & Progress Bar
        self.status_label = QLabel("", self)
        self.status_label.setStyleSheet("font-size: 14px; color: #1e293b; font-weight: 500;")
        self.status_label.setVisible(False)
        main_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                text-align: center;
                height: 14px;
                background-color: #e2e8f0;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 5px;
            }
        """)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 4. Scroll Area chứa kết quả các ảnh
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 8, 0, 8)
        self.results_layout.setSpacing(16)
        self.results_layout.addStretch()

        self.scroll_area.setWidget(self.results_container)
        main_layout.addWidget(self.scroll_area, stretch=1)

    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn ảnh nhân vật Anime",
            "",
            "Image Files (*.jpg *.jpeg *.png *.webp *.bmp);;All Files (*)",
        )
        if files:
            self.handle_files(files)

    def handle_files(self, file_paths: List[str]):
        if not file_paths:
            return

        if self.predictor is None:
            self.status_label.setText("❌ Không thể nạp mô hình. Vui lòng kiểm tra file best_arcface_model.pth!")
            self.status_label.setVisible(True)
            return

        self.status_label.setText(f"📁 Đã tải lên {len(file_paths)} bức ảnh. Hệ thống đang phân tích...")
        self.status_label.setVisible(True)
        self.progress_bar.setMaximum(len(file_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.clear_btn.setVisible(True)
        self.browse_btn.setEnabled(False)

        # Khởi chạy thread xử lý inference
        self.worker = InferenceWorker(self.predictor, file_paths)
        self.worker.progress.connect(self.update_progress)
        self.worker.image_processed.connect(self.add_image_card)
        self.worker.finished_all.connect(self.on_all_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def update_progress(self, current: int, total: int):
        self.progress_bar.setValue(current)
        self.status_label.setText(f"📁 Đang phân tích ({current}/{total} ảnh)...")

    def add_image_card(self, result: dict):
        card = ImageCardWidget(result, self)
        # Chèn thẻ trước spacer ở cuối
        count = self.results_layout.count()
        self.results_layout.insertWidget(count - 1, card)

    def on_all_finished(self):
        self.status_label.setText(f"✅ Hoàn tất phân tích tất cả các bức ảnh!")
        self.progress_bar.setVisible(False)
        self.browse_btn.setEnabled(True)

    def on_error(self, err_msg: str):
        print(f"Lỗi: {err_msg}")

    def clear_results(self):
        while self.results_layout.count() > 1:
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.status_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.clear_btn.setVisible(False)


def main():
    app = QApplication(sys.argv)
    window = AnimeClassifierApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
