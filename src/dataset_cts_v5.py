import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2

# ===== 請確認這裡的路徑正確 =====
DATA_ROOT = r"C:\AI\carpalTunnel"

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
        """
        建立索引：同時支援「分開資料夾 (MN/FT/CT)」與「單一 GT 資料夾」
        """
        self.samples.clear()
        for cid in self.case_ids:
            case_dir = os.path.join(self.root, cid)
            
            # 先判斷主要的資料夾是哪一個 (GT 或 CT)
            gt_dir = os.path.join(case_dir, "GT")
            ct_dir = os.path.join(case_dir, "CT")
            
            # 決定掃描目標資料夾
            scan_dir = gt_dir if os.path.isdir(gt_dir) else ct_dir
            if not os.path.isdir(scan_dir):
                continue

            for fname in os.listdir(scan_dir):
                if not fname.lower().endswith((".jpg", ".png", ".bmp", ".tif")):
                    continue
                
                # 建立所有可能的路徑
                t1_path = os.path.join(case_dir, "T1", fname)
                t2_path = os.path.join(case_dir, "T2", fname)
                
                # 模式 A: 單一 GT 檔案
                gt_path = os.path.join(case_dir, "GT", fname)
                
                # 模式 B: 分開的 Mask 檔案
                mn_path = os.path.join(case_dir, "MN", fname)
                ft_path = os.path.join(case_dir, "FT", fname)
                ct_path = os.path.join(case_dir, "CT", fname)

                # 檢查 T1/T2 是否存在
                if not (os.path.exists(t1_path) and os.path.exists(t2_path)):
                    continue

                # 判斷是哪種模式
                if os.path.exists(gt_path):
                    # 模式 A: 有 GT 檔
                    self.samples.append({
                        "case_id": cid, "slice_idx": fname,
                        "t1": t1_path, "t2": t2_path,
                        "mode": "single_gt",
                        "gt": gt_path
                    })
                elif os.path.exists(mn_path) and os.path.exists(ft_path) and os.path.exists(ct_path):
                    # 模式 B: 三個分開的檔 (舊資料)
                    self.samples.append({
                        "case_id": cid, "slice_idx": fname,
                        "t1": t1_path, "t2": t2_path,
                        "mode": "split_mask",
                        "mn": mn_path, "ft": ft_path, "ct": ct_path
                    })

        self.samples.sort(key=lambda x: (int(x["case_id"]), x["slice_idx"]))
        if self.debug:
            print(f"📊 Dataset 載入完成，共 {len(self.samples)} 筆資料")

    def __len__(self):
        return len(self.samples)

    def _load_gray_normalized(self, path: str) -> np.ndarray:
        """讀取 T1/T2 並做 CLAHE 增強 + 正規化"""
        img = Image.open(path).convert("L")
        arr_u8 = np.array(img, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr_enhanced = clahe.apply(arr_u8)
        return arr_enhanced.astype(np.float32) / 255.0

    def _load_single_gt(self, path):
        """
        [最終修正版] 針對 JPG 壓縮雜訊進行區間對應
        依據 check_gt.py 分析結果：
        - 數值 ~29 (Blue) -> CT (Class 3)
        - 數值 ~105 (Magenta) -> MN (Class 1)
        - 數值 ~179 (Cyan) -> FT (Class 2)
        """
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        
        # 建立空白標籤
        label = np.zeros_like(arr, dtype=np.uint8)

        # 1. 抓取 CT (腕隧道) - 數值約 29 (範圍 20~60)
        # 注意：雖然面積不一定是最大，但依據舊代碼藍色是 CT
        label[(arr >= 20) & (arr < 60)] = 3

        # 2. 抓取 FT (肌腱) - 數值約 179 (範圍 150~200)
        label[(arr >= 150) & (arr < 200)] = 2

        # 3. 抓取 MN (正中神經) - 數值約 105 (範圍 80~130)
        # MN 最重要，最後寫入以避免被覆蓋
        label[(arr >= 80) & (arr < 130)] = 1
        
        return label

    def _build_multiclass_label(self, mn_path, ft_path, ct_path):
        """舊版相容：讀取三個檔案合併"""
        img_mn = Image.open(mn_path).convert("L")
        w, h = img_mn.size
        label = np.zeros((h, w), dtype=np.uint8)

        def load_mask(path):
            if not os.path.exists(path): return np.zeros((h, w), dtype=bool)
            return np.array(Image.open(path).convert("L")) > 127

        mask_mn = load_mask(mn_path)
        mask_ft = load_mask(ft_path)
        mask_ct = load_mask(ct_path)

        label[mask_ct] = 3
        label[mask_ft] = 2
        label[mask_mn] = 1
        return label

    def random_flip_rotate(self, img, mask):
        if random.random() < 0.5:
            img = img[:, :, ::-1]
            mask = mask[:, ::-1]
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            mask = mask[::-1, :]
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2))
            mask = np.rot90(mask, k, axes=(0, 1))
        return img.copy(), mask.copy()

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 1. 讀取影像
        t1 = self._load_gray_normalized(s["t1"])
        t2 = self._load_gray_normalized(s["t2"])
        img = np.stack([t1, t2], axis=0).astype(np.float32)

        # 2. 讀取標籤
        if s["mode"] == "single_gt":
            label = self._load_single_gt(s["gt"])
        else:
            label = self._build_multiclass_label(s["mn"], s["ft"], s["ct"])

        # 3. 資料增強
        if self.augment:
            img, label = self.random_flip_rotate(img, label)

        # 4. 轉 Tensor
        img = np.ascontiguousarray(img)
        label = np.ascontiguousarray(label)

        img_tensor = torch.from_numpy(img)
        label_tensor = torch.from_numpy(label).long()

        return img_tensor, label_tensor