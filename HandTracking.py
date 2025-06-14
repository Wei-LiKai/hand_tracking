import cv2
import mediapipe as mp

# 初始化 MediaPipe 手部辨識模組
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # 最多偵測2隻手
mp_drawing = mp.solutions.drawing_utils

# 開啟攝影機
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks, handedness="Right"):
    tips = [8, 12, 16, 20]  # 食指到小指的指尖
    count = 0
    
    # 計算四根手指（食指到小指）
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    
    # 判斷拇指（根據左右手調整）
    if handedness == "Right":
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            count += 1
    else:  # Left hand
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            count += 1
    
    return count

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)  # 鏡像
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    finger_count = 0
    hand_info = []  # 儲存每隻手的資訊
    if results.multi_hand_landmarks and results.multi_handedness:
        for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            hand_label = handedness.classification[0].label
            finger_count = count_fingers(handLms, hand_label)
            hand_info.append(f"{hand_label}: {finger_count}")
            
    # 顯示手指數量
    if hand_info:
        for i, info in enumerate(hand_info):
            cv2.putText(frame, info, (20, 50 + i*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, 'No hands detected', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    cv2.imshow("Hand Tracking", frame)
    
    # 按 'q' 或 ESC 鍵退出，或點擊窗口的 X 按鈕
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 是 ESC 鍵
        break
    
    # 檢查窗口是否被關閉
    if cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
