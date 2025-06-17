import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

# 初始化 MediaPipe 手部辨識模組
mp_hands = mp.solutions.hands
# 極度降低閾值以嘗試偵測機器人手部
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.1,  # 極度降低偵測閾值
    min_tracking_confidence=0.1   # 極度降低追蹤閾值
)
mp_drawing = mp.solutions.drawing_utils

# 添加輪廓偵測功能作為備用方案
def detect_hand_contours(frame):
    """使用輪廓偵測來找到手部形狀"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 多種預處理方式
    # 方法1: 高斯模糊 + Canny邊緣偵測
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges1 = cv2.Canny(blurred, 30, 100)  # 降低閾值
    
    # 方法2: 自適應閾值
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 方法3: OTSU閾值
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 結合多種邊緣偵測結果
    edges_combined = cv2.bitwise_or(edges1, thresh1)
    edges_combined = cv2.bitwise_or(edges_combined, thresh2)
    
    # 形態學操作清理噪聲
    kernel = np.ones((3,3), np.uint8)
    edges_combined = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
    edges_combined = cv2.morphologyEx(edges_combined, cv2.MORPH_OPEN, kernel)
    
    # 找到輪廓
    contours, _ = cv2.findContours(edges_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hand_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # 大幅降低面積要求，適應不同大小的機器人手部
        if 500 < area < 100000:  # 擴大範圍
            # 檢查輪廓的長寬比，過濾掉過於細長的形狀
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # 保留合理的長寬比範圍
            if 0.3 < aspect_ratio < 3.0:
                # 計算輪廓的凸包
                hull = cv2.convexHull(contour)
                
                # 安全地計算凸缺陷，添加錯誤處理
                try:
                    hull_indices = cv2.convexHull(contour, returnPoints=False)
                    if len(hull_indices) > 3:  # 需要至少4個點才能計算凸缺陷
                        defects = cv2.convexityDefects(contour, hull_indices)
                    else:
                        defects = None
                except cv2.error:
                    # 如果計算凸缺陷失敗，設為None
                    defects = None
                  # 即使沒有凸缺陷，也添加到手部輪廓列表中
                hand_contours.append((contour, hull, defects))
    
    return hand_contours

def count_fingers_from_contour(contour, hull, defects):
    """從輪廓和凸缺陷計算手指數量"""
    if defects is None:
        return 0
    
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        
        # 計算角度
        a = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        b = ((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2) ** 0.5
        c = ((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2) ** 0.5
        
        # 使用餘弦定理計算角度
        if a > 0 and b > 0 and c > 0:
            angle = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
            if -1 <= angle <= 1:
                angle = np.arccos(angle)
                # 如果角度小於90度，可能是手指
                if angle <= np.pi / 2 and d > 30:  # d是深度
                    finger_count += 1
    
    return min(finger_count, 5)  # 最多5根手指

# 創建檔案選擇界面
def select_video_file():
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    
    # 開啟檔案選擇對話框
    file_path = filedialog.askopenfilename(
        title="選擇影片檔案",
        filetypes=[
            ("影片檔案", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("MP4 檔案", "*.mp4"),
            ("AVI 檔案", "*.avi"),
            ("MOV 檔案", "*.mov"),
            ("所有檔案", "*.*")
        ]
    )
    
    root.destroy()  # 銷毀 tkinter 根視窗
    return file_path

# 讓使用者選擇影片檔案
print("請選擇要分析的影片檔案...")
video_path = select_video_file()

if not video_path:
    print("未選擇檔案，程式結束")
    exit()

print(f"已選擇檔案: {video_path}")
cap = cv2.VideoCapture(video_path)

# 檢查影片是否成功開啟
if not cap.isOpened():
    print("無法開啟影片檔案，請檢查路徑是否正確")
    exit()

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
        print("影片播放完畢或無法讀取影片幀")
        break
        
    # 對於影片檔案，通常不需要鏡像翻轉
    # frame = cv2.flip(frame, 1)  # 如果需要鏡像可以取消註解
    
    # 嘗試圖像增強以提高機器人手部偵測
    # 調整亮度和對比度
    frame_enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
    
    # 轉換為RGB（MediaPipe需要）
    rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
      # 同時處理原始影像和增強影像
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        # 如果增強影像沒有偵測到，嘗試原始影像
        rgb_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_original)
    
    finger_count = 0
    hand_info = []  # 儲存每隻手的資訊
    mediapipe_detected = False
    
    # 先嘗試 MediaPipe 偵測
    if results.multi_hand_landmarks and results.multi_handedness:
        mediapipe_detected = True
        for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            hand_label = handedness.classification[0].label
            finger_count = count_fingers(handLms, hand_label)
            hand_info.append(f"MP-{hand_label}: {finger_count}")
      # 如果 MediaPipe 偵測不到，使用輪廓偵測作為備用方案
    if not mediapipe_detected:
        hand_contours = detect_hand_contours(frame)
        if hand_contours:
            for i, (contour, hull, defects) in enumerate(hand_contours):
                # 繪製輪廓和凸包
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
                
                # 計算手指數量
                finger_count = count_fingers_from_contour(contour, hull, defects)
                hand_info.append(f"Contour-Hand{i+1}: {finger_count}")
                
                # 標記輪廓中心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)
                    
                    # 顯示輪廓面積資訊
                    area = cv2.contourArea(contour)
                    cv2.putText(frame, f"Area: {int(area)}", (cx-50, cy-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # 如果沒有偵測到手部輪廓，顯示調試資訊
            cv2.putText(frame, 'Trying contour detection...', (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
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
