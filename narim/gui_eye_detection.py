import torch
import cv2
import numpy as np
import sys
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk

# YOLOv5 저장소 경로 추가
sys.path.append('C:\\Users\\toror\\Downloads\\hongchae-parkhyun\\yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# scale_coords 함수 정의
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords

# YOLOv5 모델 로드
model_path = "C:\\Users\\toror\\Downloads\\hongchae-parkhyun\\yolov5\\runs\\train\\exp\\weights\\best.pt"
device = select_device('cpu')
model = DetectMultiBackend(weights=str(model_path), device=device)

# Haar Cascade 로드
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# tkinter GUI 클래스
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        # tkinter 캔버스 설정
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                    height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.label = tk.Label(window, text="Hello World", width=10, height=5)
        self.label.pack()

        self.button = tk.Button(window, text="버튼", padx=15, pady=15, fg="black", bg="black", command=self.start_detection)
        self.button.pack()

        self.delay = 15
        self.detecting = False
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)  # 창 닫기 버튼(X)을 눌렀을 때 on_closing 호출
        self.window.mainloop()

    # 버튼 클릭 시 실행되는 함수
    def start_detection(self):
        self.detecting = True
        print("버튼 클릭")
        self.button.config(text="성공")

    # 웹캠에서 프레임을 읽고 업데이트하는 함수
    def update(self):
        if self.detecting:
            ret, frame = self.vid.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in eyes:
                    roi_color = frame[y:y+h, x:x+w]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    # YOLOv5를 사용하여 추가 분석 수행
                    img = cv2.resize(roi_color, (640, 640))
                    img = img[:, :, ::-1].transpose(2, 0, 1)
                    img = np.ascontiguousarray(img)

                    img = torch.from_numpy(img).to(device)
                    img = img.float()
                    img /= 255.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # 추론
                    pred = model(img, augment=False, visualize=False)
                    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

                    # 바운딩 박스 그리기 및 객체 인식 확인
                    for i, det in enumerate(pred):
                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], roi_color.shape).round()

                            for *xyxy, conf, cls in reversed(det):
                                if conf > 0.5:
                                    cv2.rectangle(roi_color, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                                    label = f"Iris: {conf:.2f}"
                                    cv2.putText(roi_color, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                # OpenCV 창에 프레임 표시
                cv2.imshow("Webcam", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.close_opencv_window(None)

        self.window.after(self.delay, self.update)

    # 'q' 키를 눌렀을 때 호출되는 함수
    def close_opencv_window(self, event):
        print("Closing OpenCV window")
        self.detecting = False
        cv2.destroyAllWindows()  # OpenCV 창 닫기

    # 창 닫기 버튼(X)을 눌렀을 때 호출되는 함수
    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        cv2.destroyAllWindows()
        self.window.destroy()

    # 객체 소멸자
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# 화면 구성
window = tk.Tk()
app = App(window, "Eye Detection with YOLOv5")
