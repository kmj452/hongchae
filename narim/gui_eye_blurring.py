import torch
import cv2
import numpy as np
import sys
from pathlib import Path
import os
import tkinter as tk

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

# Haarcascade 로드
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# YOLOv5 모델 로드
model_path = "C:\\Users\\toror\\Downloads\\hongchae-parkhyun\\yolov5\\runs\\train\\exp\\weights\\best.pt"  # 모델 경로
device = select_device('cpu')
model = DetectMultiBackend(weights=str(model_path), device=device)

# 저장할 디렉토리 설정
eye_detect_dir = "C:\\Users\\toror\\Downloads"
yolo_detect_dir = "C:\\Users\\toror\\Downloads"
if not os.path.exists(eye_detect_dir):
    os.makedirs(eye_detect_dir)
if not os.path.exists(yolo_detect_dir):
    os.makedirs(yolo_detect_dir)

# tkinter GUI 클래스
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        if not self.vid.isOpened():
            print("Error: Could not open video source.")
            sys.exit()

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
        self.frame_count = 0
        self.detect_count = 0
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
                    # 검출된 눈 영역을 확장
                    expand_ratio = 0.5  # 확장 비율
                    ex = int(w * expand_ratio)
                    ey = int(h * expand_ratio)
                    x1 = max(0, x - ex)
                    y1 = max(0, y - ey)
                    x2 = min(frame.shape[1], x + w + ex)
                    y2 = min(frame.shape[0], y + h + ey)

                    roi_color = frame[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란색 바운딩 박스 (BGR 형식으로 (255, 0, 0))

                    # 검출된 눈 영역을 이미지로 저장 (YOLO에 넣기 전)
                    eye_img_path = os.path.join(eye_detect_dir, f"eye_{self.frame_count}.png")
                    cv2.imwrite(eye_img_path, roi_color)
                    self.frame_count += 1

                    # 검출된 눈 영역을 YOLOv5 모델에 입력하기 위해 전처리
                    eye_img = cv2.resize(roi_color, (640, 640))  # YOLOv5 입력 크기로 조정
                    eye_img = eye_img[:, :, ::-1].transpose(2, 0, 1)
                    eye_img = np.ascontiguousarray(eye_img)

                    eye_img = torch.from_numpy(eye_img).to(device)
                    eye_img = eye_img.float()
                    eye_img /= 255.0
                    if eye_img.ndimension() == 3:
                        eye_img = eye_img.unsqueeze(0)

                    # 추론
                    pred = model(eye_img, augment=False, visualize=False)
                    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

                    # 바운딩 박스 그리기 및 객체 인식 확인
                    for i, det in enumerate(pred):
                        if len(det):
                            det[:, :4] = scale_coords(eye_img.shape[2:], det[:, :4], roi_color.shape).round()

                            for *xyxy, conf, cls in reversed(det):
                                if conf > 0.5:
                                    # 바운딩 박스 크기를 조정하여 홍채 부분만 포함하도록 함
                                    iris_x1, iris_y1, iris_x2, iris_y2 = map(int, xyxy)
                                    
                                    cv2.rectangle(roi_color, (iris_x1, iris_y1), (iris_x2, iris_y2), (0, 0, 255), 2)  # 빨간색 바운딩 박스 (BGR 형식으로 (0, 0, 255))
                                    label = f"Iris: {conf:.2f}"
                                    cv2.putText(roi_color, label, (iris_x1, iris_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    
                                    # 가우시안 블러 적용
                                    iris_roi = roi_color[iris_y1:iris_y2, iris_x1:iris_x2]
                                    blurred_iris_roi = cv2.GaussianBlur(iris_roi, (51, 51), 0)
                                    roi_color[iris_y1:iris_y2, iris_x1:iris_x2] = blurred_iris_roi

                                    # YOLO 검출 성공 후 이미지 저장
                                    yolo_img_path = os.path.join(yolo_detect_dir, f"detect_{self.detect_count}.png")
                                    cv2.imwrite(yolo_img_path, roi_color)
                                    self.detect_count += 1

                # OpenCV 창에 프레임 표시
                cv2.imshow("Iris Detection with YOLOv5", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.on_closing()

        self.window.after(self.delay, self.update)

    
    # 'q' 키를 눌렀을 때 호출되는 함수
    def close_opencv_window(self, event):
        print("Closing OpenCV window")
        self.detecting = False
        cv2.destroyAllWindows()  # OpenCV 창 닫기

    # 창 닫기 버튼(X)을 눌렀을 때 호출되는 함수
    def on_closing(self):
        self.detecting = False
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
