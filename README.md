# 手部追蹤手指計數器 (Hand Tracking Finger Counter)

這是一個使用 MediaPipe 和 OpenCV 進行即時手部追蹤和手指計數的 Python 程序。程序可以同時識別左右手，並準確計算每隻手伸出的手指數量。

## 功能特色

- ✅ 即時手部追蹤和識別
- ✅ 同時支援雙手識別（最多2隻手）
- ✅ 準確的手指計數（0-5根手指）
- ✅ 區分左手和右手
- ✅ 視覺化手部骨架顯示
- ✅ 多種退出方式

## 系統需求

- Python 3.7 或更高版本
- 網路攝像頭
- Windows / macOS / Linux

## 安裝教學

### 1. 克隆或下載項目

```bash
git clone <你的倉庫網址>
cd pose_detation
```

或直接下載 ZIP 檔案並解壓縮。

### 2. 創建虛擬環境（推薦）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 安裝所需套件

```bash
pip install opencv-python mediapipe
```

或者使用 requirements.txt（如果有的話）：

```bash
pip install -r requirements.txt
```

### 4. 運行程序

```bash
python test.py
```

## 使用方法

1. 運行程序後，攝像頭窗口會開啟
2. 將手放在攝像頭前方
3. 程序會自動識別左手和右手
4. 在畫面上會顯示每隻手的手指數量
5. 手部骨架會以綠色線條顯示

### 退出程序

程序提供三種退出方式：
- 按 `q` 鍵
- 按 `ESC` 鍵  
- 點擊窗口右上角的 ❌ 按鈕

## 技術說明

### 依賴套件

| 套件 | 版本 | 用途 |
|------|------|------|
| opencv-python | 最新版 | 攝像頭擷取和影像處理 |
| mediapipe | 最新版 | 手部檢測和追蹤 |

### 手指計數邏輯

程序使用 MediaPipe 提供的 21 個手部關鍵點進行手指計數：

- **四根手指（食指、中指、無名指、小指）**：比較指尖和指根的 Y 座標
- **拇指**：根據左右手分別比較 X 座標
  - 右手：拇指指尖在拇指關節左側時算作伸出
  - 左手：拇指指尖在拇指關節右側時算作伸出

### 程序設定

```python
# 最多偵測手數（1-2）
hands = mp_hands.Hands(max_num_hands=2)

# 攝像頭索引（通常是 0，如果有多個攝像頭可嘗試 1, 2...）
cap = cv2.VideoCapture(0)
```

## 常見問題

### Q: 攝像頭無法開啟
A: 嘗試修改 `cv2.VideoCapture(0)` 中的數字為 1 或 2，或檢查攝像頭是否被其他程序占用。

### Q: 手指計數不準確
A: 確保手部在光線充足的環境下，並且手部完全在攝像頭視野內。避免背景過於複雜。

### Q: 程序運行緩慢
A: 可以降低攝像頭解析度或調整 MediaPipe 的信心度閾值。

### Q: 左手識別錯誤
A: 這是正常現象，MediaPipe 有時會將鏡像後的左手識別為右手，但手指計數邏輯已經做了相應調整。

## 檔案結構

```
pose_detation/
├── test.py          # 主程序檔案
├── README.md        # 說明文件
└── requirements.txt # 依賴套件列表（可選）
```

## 授權條款

本項目使用 MIT 授權條款。詳細內容請參閱 LICENSE 檔案。

## 貢獻

歡迎提交 Issue 和 Pull Request 來改善這個項目！

## 更新日誌

### v1.0.0
- 基本手部追蹤功能
- 雙手同時識別
- 準確的手指計數
- 多種退出方式

---

如果你覺得這個項目有幫助，請給一個 ⭐ Star！

## 聯絡資訊

如有問題或建議，請通過以下方式聯絡：
- GitHub Issues: [提交問題](你的倉庫網址/issues)
- Email: [你的郵箱]
