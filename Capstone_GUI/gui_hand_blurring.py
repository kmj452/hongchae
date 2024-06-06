import tkinter as tk
import cv2
import mediapipe as mp
import math
from PIL import Image, ImageTk

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=10)

# 두 랜드마크 간의 거리 계산 함수
def calculate_distance(landmark1, landmark2, width, height):
    x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
    x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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

# tkinter GUI 클래스
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        # tkinter 캔버스 설정
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.label = tk.Label(window, text="손 블러 처리", width=10, height=5)
        self.label.pack()

        self.button = tk.Button(window, text="버튼", padx=15, pady=15, fg="white", bg="black", command=self.start_blurring)
        self.button.pack()

        self.delay = 15
        self.detecting = False

        self.window.bind('<q>', self.close_opencv_window)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def start_blurring(self):
        self.detecting = True
        print("버튼 클릭")
        self.button.config(text="성공")
        self.update_opencv()

    def update_opencv(self):
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
            self.window.after(self.delay, self.update_opencv)

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

window = tk.Tk()
app = App(window, "Hand Blurring")
