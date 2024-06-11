import cv2
import mediapipe as mp
import math
import tkinter as tk
import sys

# MediaPipe 솔루션을 초기화합니다.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

def calculate_distance(landmark1, landmark2, width, height):
    x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
    x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
    return math.sqrt((x2 - x1) ** 2 + (y1 - y2) ** 2)

def calculate_finger_length(landmarks, width, height):
    finger_lengths = []
    for finger_indices in [(4, 3, 2), (8, 7, 6), (12, 11, 10), (16, 15, 14), (20, 19, 18)]:
        finger_length = 0
        for i in range(len(finger_indices) - 1):
            finger_length += calculate_distance(landmarks[finger_indices[i]], landmarks[finger_indices[i+1]], width, height)
        finger_lengths.append(finger_length)
    return max(finger_lengths)

def blur_fingerprint_area(image, landmarks):
    h, w, _ = image.shape
    finger_length = calculate_finger_length(landmarks, w, h)
    blur_radius = int(max(15, min(50, finger_length // 4)))
    if blur_radius % 2 == 0:
        blur_radius += 1
    ksize = (blur_radius, blur_radius)
    blur_size = int(max(8, min(50, finger_length // 15)))

    for i in range(4, 21, 4):
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        x_start, y_start = max(0, x-blur_size), max(0, y-blur_size)
        x_end, y_end = min(w, x+blur_size), min(h, y+blur_size)
        if x_start < x_end and y_start < y_end:
            image[y_start:y_end, x_start:x_end] = cv2.GaussianBlur(image[y_start:y_end, x_start:x_end], ksize, 10)
    return image

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
            success, image = self.vid.read()
            if not success:
                return

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # MediaPipe Face Mesh 처리를 합니다.
            face_results = face_mesh.process(image)

            # MediaPipe Hands 처리를 합니다.
            hand_results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    iris_landmarks = [
                        face_landmarks.landmark[i] for i in range(468, 478)
                    ]
                    image = blur_iris_area(image, iris_landmarks)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    image = blur_fingerprint_area(image, hand_landmarks.landmark)

            # OpenCV 창에 프레임 표시
            cv2.imshow('MediaPipe Iris and Hand Tracking', image)

            # 'q' 키가 눌리면 on_closing 함수 호출
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.on_closing()

        self.window.after(self.delay, self.update)

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
app = App(window, "MediaPipe Iris and Hand Tracking")
