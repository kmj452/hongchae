import cv2
import mediapipe as mp
import tkinter as tk
import sys

# MediaPipe 솔루션을 초기화합니다.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

            # 성능을 향상시키기 위해 이미지를 쓰지 않고 참조합니다.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # MediaPipe Face Mesh 처리를 합니다.
            results = face_mesh.process(image)

            # 이미지에 출력을 그리기 위해 다시 이미지를 BGR로 변환합니다.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 홍채 영역을 강조하기 위해 주요 랜드마크를 표시합니다.
                    iris_landmarks = [
                        face_landmarks.landmark[i] for i in range(468, 478)
                    ]
                    for landmark in iris_landmarks:
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

            # OpenCV 창에 프레임 표시
            cv2.imshow('MediaPipe Iris Tracking', image)

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
app = App(window, "MediaPipe Iris Tracking")
