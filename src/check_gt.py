import cv2
import numpy as np
import os

# 設定成你的 GT 資料夾路徑
GT_FOLDER = r"C:\AI\carpalTunnel\testData\GT"  # <-- 請確認這路徑對不對

# 隨便抓一張圖
for f in os.listdir(GT_FOLDER):
    if f.endswith(".jpg") or f.endswith(".png"):
        path = os.path.join(GT_FOLDER, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        print(f"檢查檔案: {f}")
        print(f"圖片尺寸: {img.shape}")
        print(f"裡面的數值有: {np.unique(img)}")
        
        # 簡單統計一下每個數值的面積，通常 CT(3) 面積最大，MN(1) 面積最小
        for val in np.unique(img):
            if val == 0: continue
            area = np.sum(img == val)
            print(f"   數值 {val}: 佔用像素 {area} 點")
        
        print("-" * 30)
        break # 檢查一張就好