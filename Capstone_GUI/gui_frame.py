import tkinter as tk
import cv2
import mediapipe as mp
import math
from PIL import Image, ImageTk
import torch
import numpy as np
import sys
from pathlib import Path
import os

# YOLOv5 저장소 경로 추가
sys.path.append('C:\\Users\\toror\\Downloads\\hongchae-parkhyun\\yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=10)

# YOLOv5 모델 로드
model_path = "C:\\Users\\toror\\Downloads\\hongchae-parkhyun\\yolov5\\runs\\train\\exp\\weights\\best.pt"
device = select_device('cpu')
model = DetectMultiBackend(weights=str(model_path), device=device)

# Haar Cascade 로드
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# 두 랜드마크 간의 거리 계산 함수
def calculate_distance(landmark1, landmark2, width, height):
    x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
    x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# 손가락 끝 부분에 동그라미를 그리는 함수
def draw_finger_circles(image, landmarks):
    h, w, _ = image.shape
    thumb_tip = landmarks[4]
    pinky_tip = landmarks[20]
    distance = calculate_distance(thumb_tip, pinky_tip, w, h)

    for i in range(4, 21, 4):
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        cv2.circle(image, (x, y), 10, (0, 0, 255), 2)
    
    return image

# 손가락 끝 부분을 블러링하는 함수
def blur_fingerprint_area(image, landmarks):
    h, w, _ = image.shape
    thumb_tip = landmarks[4]
    pinky_tip = landmarks[20]
    distance = calculate_distance(thumb_tip, pinky_tip, w, h)
    
    blur_radius = int(max(15, min(50, distance // 4)))
    if blur_radius % 2 == 0:
        blur_radius += 1
    ksize = (blur_radius, blur_radius)
    
    blur_size = int(max(5, min(50, distance // 30)))
    
    for i in range(4, 21, 4):
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        x_start, y_start = max(0, x-blur_size), max(0, y-blur_size)
        x_end, y_end = min(w, x+blur_size), min(h, y+blur_size)
        if x_start < x_end and y_start < y_end:
            image[y_start:y_end, x_start:x_end] = cv2.GaussianBlur(image[y_start:y_end, x_start:x_end], ksize, 15)
    
    return image

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

# tkinter GUI 클래스
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        # 상단 텍스트 라벨
        self.label = tk.Label(window, text="인식&블러처리", font=("Arial", 30))
        self.label.pack(pady=10)

        # 버튼 프레임 설정
        self.button_frame = tk.Frame(window)
        self.button_frame.pack(pady=10)

        # 버튼 1: Hand Detection
        self.button1 = tk.Button(self.button_frame, text="Hand Detection", padx=15, pady=15, command=self.start_hand_detection)
        self.button1.grid(row=0, column=0, padx=10)

        # 버튼 2: Hand Blurring
        self.button2 = tk.Button(self.button_frame, text="Hand Blurring", padx=15, pady=15, command=self.start_hand_blurring)
        self.button2.grid(row=0, column=1, padx=10)

        # 버튼 3: Eye Detection
        self.button3 = tk.Button(self.button_frame, text="Eye Detection", padx=15, pady=15, command=self.start_eye_detection)
        self.button3.grid(row=0, column=2, padx=10)

        # 버튼 4: Eye Blurring
        self.button4 = tk.Button(self.button_frame, text="Eye Blurring", padx=15, pady=15, command=self.start_eye_blurring)
        self.button4.grid(row=0, column=3, padx=10)

        # tkinter 캔버스 설정
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.delay = 15
        self.detecting = False

        self.window.bind('<q>', self.close_opencv_window)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def start_hand_detection(self):
        self.detecting = True
        print("Hand Detection 클릭")
        self.update_opencv_hand_detection()

    def start_hand_blurring(self):
        self.detecting = True
        print("Hand Blurring 클릭")
        self.update_opencv_hand_blurring()

    def start_eye_detection(self):
        self.detecting = True
        print("Eye Detection 클릭")
        self.update_opencv_eye_detection()

    def start_eye_blurring(self):
        self.detecting = True
        print("Eye Blurring 클릭")
        self.update_opencv_eye_blurring()

    def update_opencv_hand_detection(self):
        if self.detecting:
            ret, frame = self.vid.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        frame = draw_finger_circles(frame, hand_landmarks.landmark)

                cv2.imshow("Webcam", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.close_opencv_window(None)

        if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) >= 1:
            self.window.after(self.delay, self.update_opencv_hand_detection)

    def update_opencv_hand_blurring(self):
        if self.detecting:
            ret, frame = self.vid.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        frame = blur_fingerprint_area(frame, hand_landmarks.landmark)

                cv2.imshow("Webcam", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.close_opencv_window(None)

        if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) >= 1:
            self.window.after(self.delay, self.update_opencv_hand_blurring)

    def update_opencv_eye_detection(self):
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
                                    label = f"Eye: {conf:.2f}"
                                    cv2.putText(roi_color, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Webcam", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.close_opencv_window(None)

        if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) >= 1:
            self.window.after(self.delay, self.update_opencv_eye_detection)

    def update_opencv_eye_blurring(self):
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
                    eye_img_path = os.path.join("C:\\Users\\toror\\Downloads", f"eye_{self.frame_count}.png")
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
                                    eye_x1, eye_y1, eye_x2, eye_y2 = map(int, xyxy)
                                    
                                    cv2.rectangle(roi_color, (eye_x1, eye_y1), (eye_x2, eye_y2), (0, 0, 255), 2)  # 빨간색 바운딩 박스 (BGR 형식으로 (0, 0, 255))
                                    label = f"Eye: {conf:.2f}"
                                    cv2.putText(roi_color, label, (eye_x1, eye_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    
                                    # 가우시안 블러 적용
                                    eye_roi = roi_color[eye_y1:eye_y2, eye_x1:eye_x2]
                                    blurred_eye_roi = cv2.GaussianBlur(eye_roi, (51, 51), 0)
                                    roi_color[eye_y1:eye_y2, eye_x1:eye_x2] = blurred_eye_roi

                                    # YOLO 검출 성공 후 이미지 저장
                                    yolo_img_path = os.path.join("C:\\Users\\toror\\Downloads", f"detect_{self.detect_count}.png")
                                    cv2.imwrite(yolo_img_path, roi_color)
                                    self.detect_count += 1

                cv2.imshow("Webcam", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.close_opencv_window(None)

        if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) >= 1:
            self.window.after(self.delay, self.update_opencv_eye_blurring)

    def close_opencv_window(self, event):
        if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Webcam")

    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        cv2.destroyAllWindows()
        self.window.destroy()

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# 화면 구성
window = tk.Tk()
app = App(window, "Geek Hub")
