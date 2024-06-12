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

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

def calculate_finger_length(landmarks, width, height):
    finger_lengths = []
    for finger_indices in [(4, 3, 2), (8, 7, 6), (12, 11, 10), (16, 15, 14), (20, 19, 18)]:
        finger_length = 0
        for i in range(len(finger_indices) - 1):
            finger_length += calculate_distance(landmarks[finger_indices[i]], landmarks[finger_indices[i+1]], width, height)
        finger_lengths.append(finger_length)
    return max(finger_lengths)

def blur_iris_area(image, iris_landmarks):
    h, w, _ = image.shape
    x_min = min(int(landmark.x * w) for landmark in iris_landmarks)
    y_min = min(int(landmark.y * h) for landmark in iris_landmarks)
    x_max = max(int(landmark.x * w) for landmark in iris_landmarks)
    y_max = max(int(landmark.y * h) for landmark in iris_landmarks)
    
    blur_radius = int(max(2, min(5, (x_max - x_min) // 42)))
    if blur_radius % 2 == 0:
        blur_radius += 1
    ksize = (blur_radius, blur_radius)
    x_start, y_start = max(0, x_min - blur_radius), max(0, y_min - blur_radius)
    x_end, y_end = min(w, x_max + blur_radius), min(h, y_max + blur_radius)
    
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

        if not self.vid.isOpened():
            print("Error: Could not open video source.")
            sys.exit()

        # 상단 텍스트 라벨
        self.label = tk.Label(window, text="Iris & Fingerprint Area Detection and Blurring", font=("Arial", 30))
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

        # 버튼 5: Hand & Eye Blurring
        self.button5 = tk.Button(self.button_frame, text="Hand & Eye Blurring", padx=15, pady=15, command=self.start_hand_eye_blurring)
        self.button5.grid(row=1, column=0, columnspan=4, padx=10, pady=10)

        # tkinter 캔버스 설정
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.delay = 15
        self.detecting = False
        self.frame_count = 0  # frame_count 초기화
        self.detect_count = 0  # detect_count 초기화
        self.update_webcam()

        self.window.bind('<q>', self.close_opencv_window)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update_webcam(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.window.after(self.delay, self.update_webcam)

    def start_hand_detection(self):
        self.detecting = True
        self.window.after(self.delay, self.update_opencv_hand_detection)
        print("Hand Detection 클릭")

    def start_hand_blurring(self):
        self.detecting = True
        self.window.after(self.delay, self.update_opencv_hand_blurring)
        print("Hand Blurring 클릭")

    def start_eye_detection(self):
        self.detecting = True
        self.window.after(self.delay, self.update_opencv_eye_detection)
        print("Eye Detection 클릭")

    def start_eye_blurring(self):
        self.detecting = True
        self.window.after(self.delay, self.update_opencv_eye_blurring)
        print("Eye Blurring 클릭")

    def start_hand_eye_blurring(self):
        self.detecting = True
        self.window.after(self.delay, self.update_opencv_hand_eye_blurring)
        print("Hand & Eye Blurring 클릭")

    def update_opencv_hand_detection(self):
        if self.detecting:
            ret, frame = self.vid.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        frame = draw_finger_circles(frame, hand_landmarks.landmark)

                cv2.imshow("Hand Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.close_opencv_window(None)

        if cv2.getWindowProperty("Hand Detection", cv2.WND_PROP_VISIBLE) >= 1:
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

                cv2.imshow("Hand Blurring", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.close_opencv_window(None)

        if cv2.getWindowProperty("Hand Blurring", cv2.WND_PROP_VISIBLE) >= 1:
            self.window.after(self.delay, self.update_opencv_hand_blurring)

    def update_opencv_eye_detection(self):
        if self.detecting:
            ret, frame = self.vid.read()
            if ret:
                frame.flags.writeable = False
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipe Face Mesh 처리를 합니다.
                results = face_mesh.process(frame_rgb)

                # 이미지에 출력을 그리기 위해 다시 이미지를 BGR로 변환합니다.
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # 왼쪽 눈의 랜드마크
                        left_eye_landmarks = [
                            face_landmarks.landmark[i] for i in [474, 475, 476, 477]
                        ]
                        # 오른쪽 눈의 랜드마크
                        right_eye_landmarks = [
                            face_landmarks.landmark[i] for i in [469, 470, 471, 472]
                        ]
                        
                        # 왼쪽 눈의 최소 및 최대 x, y 좌표를 구합니다.
                        min_x_left = min([int(landmark.x * frame.shape[1]) for landmark in left_eye_landmarks])
                        max_x_left = max([int(landmark.x * frame.shape[1]) for landmark in left_eye_landmarks])
                        min_y_left = min([int(landmark.y * frame.shape[0]) for landmark in left_eye_landmarks])
                        max_y_left = max([int(landmark.y * frame.shape[0]) for landmark in left_eye_landmarks])
                        
                        # 오른쪽 눈의 최소 및 최대 x, y 좌표를 구합니다.
                        min_x_right = min([int(landmark.x * frame.shape[1]) for landmark in right_eye_landmarks])
                        max_x_right = max([int(landmark.x * frame.shape[1]) for landmark in right_eye_landmarks])
                        min_y_right = min([int(landmark.y * frame.shape[0]) for landmark in right_eye_landmarks])
                        max_y_right = max([int(landmark.y * frame.shape[0]) for landmark in right_eye_landmarks])

                        # 왼쪽 눈에 빨간색 사각형으로 마킹합니다.
                        cv2.rectangle(frame, (min_x_left, min_y_left), (max_x_left, max_y_left), (0, 0, 255), 1)
                        # 오른쪽 눈에 빨간색 사각형으로 마킹합니다.
                        cv2.rectangle(frame, (min_x_right, min_y_right), (max_x_right, max_y_right), (0, 0, 255), 1)

                cv2.imshow("Eye Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.close_opencv_window(None)

        if cv2.getWindowProperty("Eye Detection", cv2.WND_PROP_VISIBLE) >= 1:
            self.window.after(self.delay, self.update_opencv_eye_detection)

    def update_opencv_eye_blurring(self):
        if self.detecting:
            ret, frame = self.vid.read()
            if ret:
                frame.flags.writeable = False
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_results = face_mesh.process(frame_rgb)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        iris_landmarks = [
                            face_landmarks.landmark[i] for i in range(468, 478)
                        ]
                        frame = blur_iris_area(frame, iris_landmarks)

                cv2.imshow("Eye Blurring", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.close_opencv_window(None)

        if cv2.getWindowProperty("Eye Blurring", cv2.WND_PROP_VISIBLE) >= 1:
            self.window.after(self.delay, self.update_opencv_eye_blurring)

    def update_opencv_hand_eye_blurring(self):
        if self.detecting:
            ret, frame = self.vid.read()
            if ret:
                frame.flags.writeable = False
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_results = face_mesh.process(frame_rgb)
                hand_results = hands.process(frame_rgb)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        iris_landmarks = [
                            face_landmarks.landmark[i] for i in range(468, 478)
                        ]
                        frame = blur_iris_area(frame, iris_landmarks)

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        frame = blur_fingerprint_area(frame, hand_landmarks.landmark)

                cv2.imshow("Hand & Eye Blurring", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.close_opencv_window(None)

        if cv2.getWindowProperty("Hand & Eye Blurring", cv2.WND_PROP_VISIBLE) >= 1:
            self.window.after(self.delay, self.update_opencv_hand_eye_blurring)

    def close_opencv_window(self, event):
        windows = ["Hand Detection", "Hand Blurring", "Eye Detection", "Eye Blurring", "Hand & Eye Blurring"]
        for win in windows:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(win)

    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        self.close_opencv_window(None)
        self.window.destroy()

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# 화면 구성
window = tk.Tk()
app = App(window, "Geek Hub")
