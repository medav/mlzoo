


from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
import torch

import cv2
import time
import sys

import torch.amp
import torch.utils.data

import torchvision.transforms as TVT
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from .. import multibox as MB
from .. import ssdrn34



cats = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]


DETECT_OBJECTS = True
FPS = 30

OUT_WIDTH = 1280
OUT_HEIGHT = 960

def pil_to_qimage(pil_img):
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG')
    q_img = QImage()
    q_img.loadFromData(buffer.getvalue())
    return q_img

class FrameGenerator(QThread):
    new_frame = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        fps = int(cap.get(5))
        print("fps:", fps)

        font = ImageFont.truetype("DejaVuSans.ttf", 24)

        net = ssdrn34.SsdResNet34()
        net.load_state_dict(torch.load('/research/data/mlmodels/ssdrn34.pth'))
        net.cuda().eval().half()

        resize = TVT.Compose([
            TVT.Resize((300, 300), antialias=True),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        t_prev = time.perf_counter()
        frame_i = 0

        while True:
            success, frame = cap.read()
            if not success: break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            print('---- Frame ----')
            print(frame.shape)

            img_pil = Image.fromarray(frame, mode='RGB')
            img_pil = img_pil.resize((OUT_WIDTH, OUT_HEIGHT))
            draw = ImageDraw.Draw(img_pil)

            draw.text((0, 0), f'Frame {frame_i}', fill=(255, 0, 0))
            frame_i += 1

            if DETECT_OBJECTS:
                # Object detection
                img = (torch.tensor(frame).cuda().permute(2, 0, 1) / 255.0).half()
                print(img.shape)
                resized = resize(img)

                with torch.no_grad():
                    dets = net.detect(resized.unsqueeze(0))[0]

                dets = [det.resize((300, 300), (OUT_HEIGHT, OUT_WIDTH)) for det in dets]
                dets : list[MB.SsdDetection]

                # THIS WORKS:
                # cv2.rectangle(frame, (103, 171), (564, 415), (255, 0, 0), 2)

                # Draw detections
                for det in dets:
                    (l, t, r, b) = det.ltrb
                    l, t, r, b = int(l), int(t), int(r), int(b)
                    print(cats[det.label], l, r, t, b)

                    draw.rectangle(((l, t), (r, b)), outline=(255, 0, 0), width=2)
                    draw.text((l, t - 30), cats[det.label], fill=(255, 0, 0), font=font)


            time.sleep(max(0, (1 / FPS) - (time.perf_counter() - t_prev)))
            t_prev = time.perf_counter()

            self.new_frame.emit(pil_to_qimage(img_pil))


class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Object Detection Stream")
        self.setGeometry(100, 100, OUT_WIDTH, OUT_HEIGHT)
        self.label = QLabel(self)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.thread = FrameGenerator()
        self.thread.new_frame.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(QImage)
    def update_image(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoWindow()
    ex.show()
    sys.exit(app.exec_())
