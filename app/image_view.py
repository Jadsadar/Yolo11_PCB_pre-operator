"""Image canvas with fit, zoom, pan, and scroll support."""
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QPainter
from PyQt6.QtWidgets import (QGraphicsScene, QGraphicsView, QHBoxLayout,
                            QLabel, QPushButton, QVBoxLayout, QWidget)


class ImageView(QWidget):
    def __init__(self, placeholder="ยังไม่มีภาพ"):
        super().__init__()
        self.placeholder = placeholder
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setMinimumSize(200, 160)
        self.item, self.fit_mode = None, True
        controls = QHBoxLayout()
        self.caption = QLabel(placeholder)
        controls.addWidget(self.caption, 1)
        for text, callback in [("−", lambda: self.zoom(.8)), ("+", lambda: self.zoom(1.25)),
                               ("พอดี", self.fit), ("1:1", self.actual_size)]:
            button = QPushButton(text)
            button.setMaximumWidth(60)
            button.clicked.connect(callback)
            controls.addWidget(button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.view, 1)

    def set_image(self, image):
        self.scene.clear()
        self.item = None
        if image is None:
            self.caption.setText(self.placeholder)
            return
        if isinstance(image, Image.Image):
            rgb = image.convert("RGB")
            raw = rgb.tobytes()
            qimage = QImage(raw, rgb.width, rgb.height, rgb.width * 3,
                            QImage.Format.Format_RGB888).copy()
        elif isinstance(image, QImage):
            qimage = image
        else:
            qimage = QImage(str(image))
        if qimage.isNull():
            self.caption.setText("ไม่พบไฟล์ภาพหรืออ่านภาพไม่ได้")
            return
        self.item = self.scene.addPixmap(QPixmap.fromImage(qimage))
        self.scene.setSceneRect(self.item.boundingRect())
        self.caption.setText(f"{qimage.width()} × {qimage.height()} px")
        self.fit()

    def clear(self):
        self.set_image(None)

    def fit(self):
        self.fit_mode = True
        if self.item:
            self.view.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor):
        if self.item and .01 <= self.view.transform().m11() * factor <= 50:
            self.fit_mode = False
            self.view.scale(factor, factor)

    def actual_size(self):
        self.fit_mode = False
        self.view.resetTransform()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit()
