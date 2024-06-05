#GUI 환경에서 OpenCV를 활용한 웹캠이 잘 실행되는지 테스트하는 코드
import tkinter as tk
import cv2

#실행 함수
def open_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam", frame)

        key = cv2.waitkey(2)
        if key == 27:  #ESC: 종료
            break

    cap.release()
    cv2.destroyAllWindows()

def button_click():
    print("버튼 클릭")
    button.config(text="성공")
    open_webcam()

#화면 구성
window = tk.Tk()

window.title("Hand Detection")
window.geometry("500x500+100+100")
window.resizable(True, True)

label = tk.Label(window, text="Hello World", width=10, height=5)
label.pack()

button = tk.Button(window, text="버튼", padx=15, pady=15, fg="black", bg="black", command=button_click)
button.pack()

window.mainloop()