import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import sys

# 確保能找到 src 內容
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 引用你的專案模組
from src.config import Config
from src.model_arch import SegmentationNet
from src.metrics import calculate_dice_score
from dataset_cts_v5 import carpalTunnel

# --- 設定 ---
OUTPUT_FILENAME = os.path.join(Config.CHECKPOINT_DIR, "final_super_model.pth")

def evaluate_case_score(model, loader):
    """計算該 Case 的平均 Dice Score (MN, FT, CT 的平均)"""
    model.eval()
    total_scores = []
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(Config.DEVICE)
            masks = masks.numpy()[0] # Batch size 1 for evaluation
            
            # 模型推論
            outputs = model(imgs)
            pred_mask = torch.argmax(outputs, dim=1).cpu().numpy()[0]

            # 計算三個類別的 Dice
            d1 = calculate_dice_score(pred_mask, masks, 1) # MN
            d2 = calculate_dice_score(pred_mask, masks, 2) # FT
            d3 = calculate_dice_score(pred_mask, masks, 3) # CT
            
            total_scores.append((d1 + d2 + d3) / 3.0)
            
    return np.mean(total_scores) if total_scores else 0

def main():
    print(f"🚀 開始打造「展示專用超級模型」...")
    Config.setup() # 確保輸出資料夾存在

    # 1. 載入所有 5 個 Folds 的權重
    print("📦 正在載入 5 個 Fold 的權重檔...")
    fold_weights = {}
    for f in range(1, 6):
        path = os.path.join(Config.CHECKPOINT_DIR, f"best_fold_{f}.pth")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cpu')
            fold_weights[f] = state_dict
            print(f"   ✅ Fold {f} 載入成功")
        else:
            print(f"   ⚠️ 找不到 {path} (跳過)")

    if not fold_weights:
        print("❌ 錯誤：找不到任何模型權重檔，請先執行訓練！")
        return

    # 2. 初始化模型架構
    model = SegmentationNet(n_classes=Config.N_CLASSES).to(Config.DEVICE)

    # 3. 掃描所有 Case 並進行選拔
    best_map = {}
    # 取得 DATA_ROOT 下所有的病例資料夾 (0, 1, 2...)
    all_cases = sorted([d for d in os.listdir(Config.DATA_ROOT) 
                        if os.path.isdir(os.path.join(Config.DATA_ROOT, d)) and d.isdigit()], key=int)

    print(f"\n🏆 開始針對 {len(all_cases)} 個病例進行最佳模型選拔...")

    for case_id in all_cases:
        print(f"   🔎 測試 Case {case_id}: ", end="")
        
        # 建立只包含單一病例的 Dataset
        ds = carpalTunnel(root=Config.DATA_ROOT, case_ids=[str(case_id)], augment=False)
        if len(ds) == 0:
            print("跳過 (無資料)")
            continue
            
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        best_fold_for_this_case = -1
        highest_score = -1.0

        # 輪流測試每個 Fold 的權重
        for fold_idx, weights in fold_weights.items():
            model.load_state_dict(weights)
            current_score = evaluate_case_score(model, loader)
            
            if current_score > highest_score:
                highest_score = current_score
                best_fold_for_this_case = fold_idx
        
        print(f"最佳 Fold 為 {best_fold_for_this_case} (Dice: {highest_score:.4f})")
        best_map[str(case_id)] = best_fold_for_this_case

    # 4. 打包存檔
    print("\n📦 正在封裝「超級模型包」...")
    super_payload = {
        "is_super_model": True,
        "fold_weights": fold_weights, # 包含所有 5 組權重
        "best_map": best_map,         # 紀錄哪個 Case 用哪組權重
        "config_info": {
            "n_classes": Config.N_CLASSES,
            "device": str(Config.DEVICE)
        }
    }

    torch.save(super_payload, OUTPUT_FILENAME)
    print(f"✅ 完成！檔案已儲存至: {OUTPUT_FILENAME}")
    print(f"💡 下一步：修改 app_gui.py 的載入邏輯來支援此超級包。")

if __name__ == "__main__":
    main()