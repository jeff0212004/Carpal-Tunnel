import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2  # ★★★ 核心武器：OpenCV 用於 CLAHE

# ===== 請確認這裡的路徑正確 =====
DATA_ROOT = r"C:\AI\carpalTunnel"

# 類別 ID 對應
# 0 = 背景, 1 = MN, 2 = FT, 3 = CT
CLASS_IDS = {
    "MN": 1,
    "FT": 2,
    "CT": 3,
}

def _load_gray_normalized(path: str) -> np.ndarray:
    """
    讀取灰階圖 (T1/T2) 並做 CLAHE 增強 + 正規化。
    這是讓 MN/FT 分數暴漲的關鍵。
    """
    # 1. 讀取原始影像 (0-255)
    img = Image.open(path).convert("L")
    arr_u8 = np.array(img, dtype=np.uint8)
    
    # ★★★ 應用 CLAHE (對比度增強) ★★★
    # clipLimit=2.0 讓細節浮現，但不過度曝光
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr_enhanced = clahe.apply(arr_u8)
    
    # 3. 正規化到 0.0 ~ 1.0
    arr_float = arr_enhanced.astype(np.float32) / 255.0
    return arr_float

def _load_mask_by_color(path: str):
    """
    修正版：直接從彩色圖中根據 RGB 數值提取標籤，不轉灰階。
    """
    # 1. 讀取 RGB 影像 (不轉 L)
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    # 2. 針對你上傳的 0.jpg 顏色進行精準抓取
    # 粉紅色 (MN): 紅、藍 通道都很高
    mn_mask = (r > 150) & (b > 150) & (g < 100)
    
    # 青色 (FT): 綠、藍 通道都很高
    ft_mask = (g > 150) & (b > 150) & (r < 100)
    
    # 深藍色 (CT): 只有 藍 通道高
    ct_mask = (b > 100) & (r < 100) & (g < 100)
    
    # ... 根據你的 _build_multiclass_label 邏輯組合這些 mask

def random_flip_rotate(img: np.ndarray, mask: np.ndarray):
    """
    資料增強：隨機翻轉與旋轉 (訓練時增加多樣性)
    """
    # 水平翻轉
    if random.random() < 0.5:
        img = img[:, :, ::-1]
        mask = mask[:, ::-1]

    # 垂直翻轉
    if random.random() < 0.5:
        img = img[:, ::-1, :]
        mask = mask[::-1, :]

    # 隨機旋轉 0/90/180/270
    k = random.randint(0, 3)
    if k > 0:
        img = np.rot90(img, k, axes=(1, 2))
        mask = np.rot90(mask, k, axes=(0, 1))

    return img, mask

class carpalTunnel(Dataset):
    def __init__(
        self,
        root: str = DATA_ROOT,
        case_ids=None,
        augment: bool = False,
        debug=False
    ):
        super().__init__()
        self.root = root
        self.augment = augment
        self.debug = debug

        # 掃描 case (0~9)
        if case_ids is None:
            case_ids = [
                d for d in os.listdir(root)
                if d.isdigit() and os.path.isdir(os.path.join(root, d))
            ]
            case_ids = sorted(case_ids, key=lambda x: int(x))
        self.case_ids = case_ids

        self.samples = []
        self._build_index()

    def _build_index(self):
        self.samples.clear()
        for cid in self.case_ids:
            case_dir = os.path.join(self.root, cid)
            # 以 CT 資料夾作為基準來找檔案清單
            base_dir = os.path.join(case_dir, "CT") 
            
            if not os.path.isdir(base_dir):
                continue

            for fname in os.listdir(base_dir):
                if not fname.lower().endswith((".jpg", ".png")):
                    continue
                
                # 組合所有路徑
                ct_path = os.path.join(case_dir, "CT", fname)
                t1_path = os.path.join(case_dir, "T1", fname) # 確保讀取 T1 資料夾
                t2_path = os.path.join(case_dir, "T2", fname) # 確保讀取 T2 資料夾
                mn_path = os.path.join(case_dir, "MN", fname)
                ft_path = os.path.join(case_dir, "FT", fname)

                # 確保 T1, T2, MN 這些核心檔案都存在才加入訓練
                if os.path.exists(t1_path) and os.path.exists(t2_path) and os.path.exists(mn_path):
                    self.samples.append({
                        "case_id": cid,
                        "slice_idx": fname,
                        "t1": t1_path,  # 存入小寫 t1
                        "t2": t2_path,  # 存入小寫 t2
                        "ct": ct_path,
                        "mn": mn_path,
                        "ft": ft_path
                    })

        # 排序：這對 GUI 很重要，訓練時沒差但保持一致比較好
        self.samples.sort(key=lambda x: (int(x["case_id"]), x["slice_idx"]))

    def __len__(self):
        return len(self.samples)

    def _build_multiclass_label(self, mn_path, ft_path, ct_path):
        """
        修正版：分別讀取三個獨立資料夾的遮罩並合併
        假設遮罩檔是黑底白字 (0與255) 或灰階圖
        """
        # 1. 讀取第一張圖來確定尺寸
        img_mn = Image.open(mn_path).convert("L")
        w, h = img_mn.size
        
        # 初始化空白標籤 (背景=0)
        label = np.zeros((h, w), dtype=np.uint8)

        # 定義讀取函式：讀取圖片 -> 轉 Array -> 閾值二值化
        def load_mask(path):
            if not os.path.exists(path):
                return np.zeros((h, w), dtype=bool)
            # 讀取並轉為 numpy
            m = np.array(Image.open(path).convert("L"))
            # 設定閾值 (大於 127 當作有東西)
            return m > 127

        # 2. 分別讀取三個遮罩
        mask_mn = load_mask(mn_path)
        mask_ft = load_mask(ft_path)
        mask_ct = load_mask(ct_path)

        # 3. 合併標籤 (注意順序：後面的會覆蓋前面的)
        # 建議順序：範圍最大的在最底層，範圍最小的在最上層
        # 通常 CT(腕隧道) 最大，FT(肌腱) 次之，MN(神經) 最重要
        
        # 填入 CT (Class 3)
        label[mask_ct] = 3
        
        # 填入 FT (Class 2)
        label[mask_ft] = 2
        
        # 填入 MN (Class 1) - 確保神經不會被覆蓋
        label[mask_mn] = 1

        return label

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 1. 讀取影像 (Input)
        t1 = _load_gray_normalized(s["t1"])
        t2 = _load_gray_normalized(s["t2"])
        img = np.stack([t1, t2], axis=0).astype(np.float32)

        # 2. 讀取標籤 (使用新修正的函式)
        label = self._build_multiclass_label(s["mn"], s["ft"], s["ct"])

        # 3. 資料增強
        if self.augment:
            img, label = random_flip_rotate(img, label)

        # 4. 轉 Tensor
        img = np.ascontiguousarray(img)
        label = np.ascontiguousarray(label)

        img_tensor = torch.from_numpy(img)
        label_tensor = torch.from_numpy(label).long()

        return img_tensor, label_tensor