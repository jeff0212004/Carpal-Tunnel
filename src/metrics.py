import numpy as np

def calculate_dice_score(pred_mask, gt_mask, class_id):
    """
    計算特定類別的 Dice Coefficient
    針對 CT (class_id=3) 進行特殊的聯集處理 (Union Logic)
    """
    # 邏輯判斷
    if class_id == 3: 
        # CT 定義為所有組織的聯集 (假設 CT 包含內容物)
        pred_bin = (pred_mask == 1) | (pred_mask == 2) | (pred_mask == 3)
        gt_bin = (gt_mask == 1) | (gt_mask == 2) | (gt_mask == 3)
    else:
        # 一般類別直接比對
        pred_bin = (pred_mask == class_id)
        gt_bin = (gt_mask == class_id)

    # 計算交集與聯集
    intersection = (pred_bin & gt_bin).sum()
    total_area = pred_bin.sum() + gt_bin.sum()

    if total_area == 0:
        return 1.0  # 若兩者皆無此組織，視為完美預測
    
    return 2.0 * intersection / (total_area + 1e-6)