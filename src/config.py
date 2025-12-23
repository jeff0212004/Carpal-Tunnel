import torch
import os

class Config:
    # --- 路徑設定 ---
    # 這裡將模型儲存資料夾改個名字，更有個人風格
    CHECKPOINT_DIR = "experiments_output" 
    DATA_ROOT = r"C:\AI\carpalTunnel" # 若有變更請修改這裡
    
    # --- 硬體設定 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 訓練超參數 ---
    N_CLASSES = 4
    BATCH_SIZE = 4      # 根據顯卡記憶體調整，原設1可能太小
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    
    # --- 類別權重 (MN, FT, CT) ---
    # 對應背景(0), MN(1), FT(2), CT(3)
    CLASS_WEIGHTS = [0.5, 10.0, 2.0, 2.0]

    @staticmethod
    def setup():
        """自動建立輸出資料夾"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)