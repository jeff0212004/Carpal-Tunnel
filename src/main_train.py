import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
import os

# --- 修正重點 1: 正確引用模組 ---
# 確保 Python 能找到 src 資料夾
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 引用 src 裡的設定與架構
from src.config import Config
from src.model_arch import SegmentationNet
from src.engine import Trainer

# --- 修正重點 2: 正確引用資料集 ---
# dataset_cts_v5.py 在同一層目錄，類別名稱是 carpalTunnel
try:
    from dataset_cts_v5 import carpalTunnel
except ImportError:
    print("❌ 錯誤: 找不到 dataset_cts_v5.py")
    print("   請確認該檔案是否與 main_train.py 放在同一個資料夾中！")
    sys.exit(1)

def run_kfold_training():
    # 1. 建立輸出資料夾
    Config.setup()
    
    # 定義 5-Fold 分組 (資料集編號)
    pairs = [["8", "9"], ["6", "7"], ["4", "5"], ["2", "3"], ["0", "1"]]
    
    print(f"🔥 開始執行 5-Fold 交叉驗證 | Device: {Config.DEVICE}")
    print(f"📂 資料讀取路徑: {Config.DATA_ROOT}")

    for fold_idx in range(5):
        val_ids = [pairs[fold_idx][0]]
        test_ids = [pairs[fold_idx][1]] # 這裡我們暫時把 test 當 val 用，或者你可以保留 test 不參與訓練
        
        # 建立訓練清單 (排除驗證集跟測試集)
        train_ids = []
        for p in pairs:
            if p != pairs[fold_idx]:
                train_ids.extend(p)
                
        print(f"\n=== Fold {fold_idx+1}/5 | Train: {train_ids} | Val: {val_ids} ===")

        # 2. 準備資料集
        # 注意：如果你的 carpalTunnel 不支援 augment 參數，請將其刪除或改為 is_train=True
        try:
            train_ds = carpalTunnel(root=Config.DATA_ROOT, case_ids=train_ids, augment=True)
            val_ds = carpalTunnel(root=Config.DATA_ROOT, case_ids=val_ids, augment=False)
        except TypeError:
            print("⚠️ 警告: 你的 Dataset 可能不支援 'augment' 參數，正在嘗試使用預設設定...")
            train_ds = carpalTunnel(root=Config.DATA_ROOT, case_ids=train_ids)
            val_ds = carpalTunnel(root=Config.DATA_ROOT, case_ids=val_ids)

        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
        # 驗證集 batch_size 設為 1 以便計算 Dice
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        # 3. 初始化模型與優化器
        model = SegmentationNet(n_classes=Config.N_CLASSES).to(Config.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        
        # 學習率調整策略: 當驗證分數不升反降時，降低學習率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        # 4. 訓練流程引擎
        trainer = Trainer(model, optimizer, scheduler, train_loader, val_loader)
        
        best_score = 0
        patience_counter = 0
        
        for epoch in range(1, Config.EPOCHS + 1):
            loss = trainer.train_epoch(epoch)
            val_score = trainer.validate()
            
            # 更新學習率
            scheduler.step(val_score)

            print(f"   Ep {epoch} | Loss: {loss:.4f} | Val Score: {val_score:.4f}")

            # 儲存最佳模型
            if val_score > best_score:
                best_score = val_score
                save_path = f"{Config.CHECKPOINT_DIR}/best_fold_{fold_idx+1}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"   >>> 🏆 Model Saved: {save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early Stopping (早停機制)
            if patience_counter >= 15:
                print("🛑 Early Stopping triggered (模型分數未再提升，提早結束)")
                break

if __name__ == "__main__":
    run_kfold_training()