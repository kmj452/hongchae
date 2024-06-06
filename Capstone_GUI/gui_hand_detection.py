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

# tkinter GUI 클래스
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        # tkinter 캔버스 설정
        self.canvas = tk.Canvas(window, width=640, height=480)  # 고정된 크기의 캔버스
        self.canvas.pack()

        self.label = tk.Label(window, text="손 감지", width=10, height=5)
        self.label.pack()

        self.button = tk.Button(window, text="버튼", padx=15, pady=15, fg="white", bg="black", command=self.start_detection)
        self.button.pack()

        self.delay = 15
        self.detecting = False

        self.window.bind('<q>', self.close_opencv_window)  # 'q' 키를 눌렀을 때 OpenCV 창 닫기
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)  # 창 닫기 버튼(X)을 눌렀을 때 on_closing 호출
        self.window.mainloop()

    # 버튼 클릭 시 실행되는 함수
    def start_detection(self):
        self.detecting = True
        print("버튼 클릭")
        self.button.config(text="성공")
        self.update_opencv()

    # OpenCV 창을 업데이트하는 함수
    def update_opencv(self):
        if self.detecting:
            ret, frame = self.vid.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        frame = draw_finger_circles(frame, hand_landmarks.landmark)

                # OpenCV 창에 프레임 표시
                cv2.imshow("Webcam", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 눌렀을 때 창 닫기
                    self.close_opencv_window(None)

        if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) >= 1:
            self.window.after(self.delay, self.update_opencv)  # OpenCV 창이 열려 있으면 업데이트 계속

    # 'q' 키를 눌렀을 때 호출되는 함수
    def close_opencv_window(self, event):
        if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Webcam")  # OpenCV 창 닫기

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
app = App(window, "Hand Detection")
