import tkinter as tk
#import cv2
#import mediapipe as mp
#import math
#from PIL import Image, ImageTk

# tkinter GUI 클래스
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # 상단 텍스트 라벨
        self.label = tk.Label(window, text="인식&블러처리", font=("Arial", 30))
        self.label.pack(pady=10)

        # 버튼 프레임 설정
        self.button_frame = tk.Frame(window)
        self.button_frame.pack(pady=10)

        # 버튼 1
        self.button1 = tk.Button(self.button_frame, text="Hand Detection", padx=15, pady=15, command=self.button1_action)
        self.button1.grid(row=0, column=0, padx=10)

        # 버튼 2
        self.button2 = tk.Button(self.button_frame, text="Hand Blurring", padx=15, pady=15, command=self.button2_action)
        self.button2.grid(row=0, column=1, padx=10)

        # 버튼 3
        self.button3 = tk.Button(self.button_frame, text="Eye Detection", padx=15, pady=15, command=self.button3_action)
        self.button3.grid(row=0, column=2, padx=10)

        # 버튼 4
        self.button4 = tk.Button(self.button_frame, text="Eye Blurring", padx=15, pady=15, command=self.button4_action)
        self.button4.grid(row=0, column=3, padx=10)

    def button1_action(self):
        print("Hand Detection 클릭")

    def button2_action(self):
        print("Hand Blurring 클릭")

    def button3_action(self):
        print("Eye Detection 클릭")

    def button4_action(self):
        print("Eye Blurring 클릭")

# 화면 구성
window = tk.Tk()
app = App(window, "Geek Hub")
window.mainloop()
