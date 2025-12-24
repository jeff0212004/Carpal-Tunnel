import sys
import os
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QFrame, QGridLayout, QMessageBox, QComboBox, 
                             QSplitter, QProgressBar)
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor
from PyQt6.QtCore import Qt

# 引用 src 模組
# 確保你的資料夾結構中有 src/config.py 和 src/model_arch.py
from src.config import Config
from src.model_arch import SegmentationNet

# 引用資料集設定
try:
    from dataset_cts_v5 import DATA_ROOT as DEFAULT_DATA_ROOT
except ImportError:
    DEFAULT_DATA_ROOT = Config.DATA_ROOT

class CTSInspector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical AI Segmentation System - Final Project")
        self.resize(1500, 850) 
        self.init_style()

        # 1. 核心變數初始化
        self.is_super_mode = False
        self.super_weights = {}
        self.case_to_fold_map = {}
        self.current_model_path = None
        self.image_list = []
        self.t1_folder = ""
        self.t2_folder = ""
        self.gt_folders = {} 
        self.root_dir = DEFAULT_DATA_ROOT

        # 2. 初始化模型架構
        self.seg_net = SegmentationNet(n_classes=Config.N_CLASSES).to(Config.DEVICE)
        self.seg_net.eval()
        
        # 3. 建立 UI
        self.setup_ui()
        
        # 4. 掃描現有模型與資料
        self.scan_for_models()

    def imread_safe(self, path, grayscale=True):
        """
        強力讀取函式：解決 Windows 中文路徑問題
        """
        try:
            if not os.path.exists(path):
                return None
            stream = open(path, "rb")
            bytes_data = bytearray(stream.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            return cv2.imdecode(numpyarray, mode)
        except Exception as e:
            print(f"讀取錯誤: {path} | Error: {e}")
            return None

    def init_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f7fa; }
            QFrame#MainPanel { 
                background-color: white; 
                border-radius: 10px; 
                border: 1px solid #e0e0e0;
            }
            QLabel { font-family: 'Segoe UI', 'Microsoft JhengHei'; color: #333333; }
            QLabel#Title { color: #2c3e50; font-weight: bold; }
            QPushButton { 
                background-color: #3498db; color: white; border: none; 
                padding: 10px 20px; border-radius: 5px; font-weight: bold; 
            }
            QPushButton:hover { background-color: #2980b9; }
            QComboBox { padding: 5px; border: 1px solid #bdc3c7; border-radius: 4px; background-color: white; }
            QSlider::groove:horizontal { border: 1px solid #999999; height: 8px; background: white; border-radius: 4px; }
            QSlider::handle:horizontal { background: #3498db; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; }
        """)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # --- 左側控制板 ---
        control_panel = QFrame()
        control_panel.setObjectName("MainPanel")
        control_panel.setFixedWidth(350)
        layout.addWidget(control_panel)
        vbox = QVBoxLayout(control_panel)
        vbox.setSpacing(15)
        
        title = QLabel("AI 影像分析儀")
        title.setObjectName("Title")
        title.setFont(QFont("Microsoft JhengHei", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(title)

        vbox.addWidget(QLabel("1. 選擇模型權重:"))
        self.btn_load_model = QPushButton("📂 載入模型 (.pth)")
        self.btn_load_model.clicked.connect(self.load_model)
        vbox.addWidget(self.btn_load_model)
        
        self.lbl_status = QLabel("狀態: 尚未載入模型")
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        vbox.addWidget(self.lbl_status)
        
        vbox.addWidget(QLabel("2. 選擇病例 (Case):"))
        self.combo_case = QComboBox()
        self.combo_case.currentTextChanged.connect(self.on_case_selected)
        vbox.addWidget(self.combo_case)

        vbox.addSpacing(20)
        vbox.addWidget(QLabel("當前切片 Dice 分數:"))
        self.score_container = QFrame()
        self.score_container.setStyleSheet("background-color: #f9f9f9; border-radius: 5px;")
        score_layout = QVBoxLayout(self.score_container)
        
        self.lbl_scores = {}
        metrics_config = [("MN", "正中神經", "#f1c40f"), ("FT", "屈肌腱", "#3498db"), ("CT", "腕隧道", "#e74c3c")]

        for key, name, color in metrics_config:
            h_layout = QHBoxLayout()
            name_lbl = QLabel(f"• {name}:")
            name_lbl.setStyleSheet(f"font-weight: bold; color: {color};")
            score_lbl = QLabel("0.00")
            score_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            score_lbl.setStyleSheet(f"font-weight: bold; color: {color}; background-color: white; border: 1px solid {color}; padding: 2px;")
            h_layout.addWidget(name_lbl)
            h_layout.addWidget(score_lbl)
            score_layout.addLayout(h_layout)
            self.lbl_scores[key] = score_lbl
            
        vbox.addWidget(self.score_container)
        vbox.addStretch()

        # --- 右側顯示區 ---
        display_area = QFrame()
        display_area.setObjectName("MainPanel")
        layout.addWidget(display_area)
        r_layout = QVBoxLayout(display_area)

        img_headers = QHBoxLayout()
        for name in ["原始 T1 影像", "真實標註 (GT)", "AI 預測結果"]:
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFont(QFont("Microsoft JhengHei", 11, QFont.Weight.Bold))
            img_headers.addWidget(lbl)
        r_layout.addLayout(img_headers)

        img_layout = QHBoxLayout()
        self.views = {}
        for key in ["input", "gt", "pred"]:
            lbl_img = QLabel()
            lbl_img.setFixedSize(360, 360) 
            lbl_img.setStyleSheet("background-color: black; border-radius: 4px;")
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_layout.addWidget(lbl_img)
            self.views[key] = lbl_img
        r_layout.addLayout(img_layout)
        
        control_bar = QHBoxLayout()
        control_bar.addWidget(QLabel("切片索引:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.update_view)
        control_bar.addWidget(self.slider)
        self.lbl_idx = QLabel("0/0")
        control_bar.addWidget(self.lbl_idx)
        r_layout.addLayout(control_bar)

    def scan_for_models(self):
        if os.path.exists(self.root_dir):
            self.combo_case.clear() 
            cases = []
            for d in os.listdir(self.root_dir):
                if os.path.isdir(os.path.join(self.root_dir, d)):
                    cases.append(d)
            try:
                cases.sort(key=lambda x: int(x) if x.isdigit() else x)
            except:
                cases.sort()
            self.combo_case.addItems(cases)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇模型檔案", Config.CHECKPOINT_DIR, "Model Files (*.pth)")
        if file_path:
            try:
                checkpoint = torch.load(file_path, map_location=Config.DEVICE)
                if isinstance(checkpoint, dict) and checkpoint.get("is_super_model"):
                    self.is_super_mode = True
                    self.super_weights = checkpoint["fold_weights"]
                    self.case_to_fold_map = checkpoint["best_map"]
                    current_case = self.combo_case.currentText()
                    if current_case in self.case_to_fold_map:
                        f_idx = self.case_to_fold_map[current_case]
                        self.seg_net.load_state_dict(self.super_weights[f_idx])
                        self.lbl_status.setText(f"狀態: 超級模式 (Case {current_case} -> Fold {f_idx})")
                    QMessageBox.information(self, "成功", f"✨ 已載入超級模型包！")
                else:
                    self.is_super_mode = False
                    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
                    self.seg_net.load_state_dict(state_dict)
                    QMessageBox.information(self, "成功", "已載入一般權重。")
                
                self.current_model_path = file_path
                self.lbl_status.setText(f"當前模型: {os.path.basename(file_path)}")
                if self.image_list: self.update_view(self.slider.value())
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"載入模型失敗: {str(e)}")

    def on_case_selected(self, case_id):
        if self.is_super_mode and case_id in self.case_to_fold_map:
            best_fold = self.case_to_fold_map[case_id]
            if best_fold in self.super_weights:
                self.seg_net.load_state_dict(self.super_weights[best_fold])
                self.lbl_status.setText(f"狀態: 超級模式 (Case {case_id} -> Fold {best_fold})")
        
        case_path = os.path.join(self.root_dir, case_id)
        self.load_case_data(case_path)

    def load_case_data(self, case_path=None):
        if case_path is None:
            case_id = self.combo_case.currentText()
            if not case_id: return
            case_path = os.path.join(self.root_dir, case_id)
        
        self.t1_folder = os.path.join(case_path, "T1")
        self.t2_folder = os.path.join(case_path, "T2")
        print(f"\n=== 正在載入病例: {os.path.basename(case_path)} ===")
        
        # 優先權判斷：先看有沒有分開的 MN/FT/CT 資料夾，如果沒有就看有沒有 GT 資料夾
        path_mn = os.path.join(case_path, "MN")
        path_gt_single = os.path.join(case_path, "GT")
        
        # 有些情況 GT 資料夾可能叫 CT，做個防呆
        path_ct_as_gt = os.path.join(case_path, "CT")
        
        self.gt_mode = "none" # 預設

        if os.path.exists(path_mn):
            self.gt_mode = "split"
            self.gt_folders = {
                1: os.path.join(case_path, "MN"),
                2: os.path.join(case_path, "FT"),
                3: os.path.join(case_path, "CT")
            }
            print("模式偵測: 分離版 MN/FT/CT")
            
        elif os.path.exists(path_gt_single):
            self.gt_mode = "single"
            self.gt_folder = path_gt_single
            print("模式偵測: 整合版 GT")
            
        elif os.path.exists(path_ct_as_gt) and not os.path.exists(path_mn):
            # 只有 CT 資料夾，但裡面其實是整合的 GT
            self.gt_mode = "single"
            self.gt_folder = path_ct_as_gt
            print("模式偵測: 整合版 GT (在 CT 資料夾內)")

        if os.path.exists(self.t1_folder):
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
            self.image_list = sorted(
                [f for f in os.listdir(self.t1_folder) if f.lower().endswith(valid_exts)],
                key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
            )
            self.slider.setMaximum(max(0, len(self.image_list) - 1))
            self.slider.setEnabled(len(self.image_list) > 0)
            self.slider.setValue(0)
            if self.image_list: self.update_view(0)

    def predict(self, img_name):
        p1, p2 = os.path.join(self.t1_folder, img_name), os.path.join(self.t2_folder, img_name)
        i1 = self.imread_safe(p1, grayscale=True)
        i2 = self.imread_safe(p2, grayscale=True)
        if i1 is None or i2 is None: return None, None
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        i1_c, i2_c = clahe.apply(i1), clahe.apply(i2)
        inp = np.stack([i1_c, i2_c], axis=0).astype(np.float32) / 255.0
        inp_t = torch.from_numpy(inp).unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            logits = self.seg_net(inp_t)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
        return pred, i1

    def get_gt(self, img_name):
        """
        [修正版] 依據 check_gt.py 的結果進行數值對應
        """
        mask = np.zeros((512, 512), dtype=np.uint8)
        base_name = os.path.splitext(img_name)[0]

        def find_and_load(folder, target_base_name):
            # 嘗試直接用檔名
            direct_path = os.path.join(folder, img_name)
            img = self.imread_safe(direct_path, grayscale=True)
            if img is not None: return img
            
            # 嘗試用 base_name 搜尋 (避免 jpg vs png 問題)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if os.path.splitext(f)[0] == target_base_name:
                        return self.imread_safe(os.path.join(folder, f), grayscale=True)
            return None

        # 處理 Single GT 模式 (新的 GT 資料夾)
        if hasattr(self, "gt_mode") and self.gt_mode == "single":
            gt_img = find_and_load(self.gt_folder, base_name)
            if gt_img is not None:
                max_v = np.max(gt_img)
                
                # 狀況 A: 已經是 0,1,2,3 的格式
                if max_v <= 3 and max_v > 0:
                    mask = gt_img
                # 狀況 B: 是灰階圖 (JPG壓縮過)，需要區間對應
                elif max_v > 3:
                    # 依據 check_gt.py 的分析結果：
                    # 數值 ~29 (Blue) -> CT (Class 3)
                    mask[(gt_img >= 20) & (gt_img < 60)] = 3
                    
                    # 數值 ~179 (Cyan) -> FT (Class 2)
                    mask[(gt_img >= 150) & (gt_img < 200)] = 2
                    
                    # 數值 ~105 (Magenta) -> MN (Class 1)
                    mask[(gt_img >= 80) & (gt_img < 130)] = 1
            
        # 處理 Split 模式 (舊的 MN/FT/CT 資料夾)
        elif hasattr(self, "gt_mode") and self.gt_mode == "split":
            t1_img = find_and_load(self.t1_folder, base_name)
            h, w = t1_img.shape if t1_img is not None else (512, 512)
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # 依序讀取 (CT -> FT -> MN，讓小的蓋在大的上面)
            for idx, key in [(3, 3), (2, 2), (1, 1)]:
                part = find_and_load(self.gt_folders[idx], base_name)
                if part is not None: 
                    mask[part > 127] = key
            
        return mask

    def update_view(self, val):
        if not self.image_list: return
        fname = self.image_list[val]
        self.lbl_idx.setText(f"{val+1}/{len(self.image_list)}")
        
        pred_mask, t1_img = self.predict(fname)
        gt_mask = self.get_gt(fname)
        
        if pred_mask is not None:
            # 這裡需要引用 calculate_dice_score，若無 metrics.py 可用可忽略
            try:
                from src.metrics import calculate_dice_score
                for k, idx in [("MN", 1), ("FT", 2), ("CT", 3)]:
                    score = calculate_dice_score(pred_mask, gt_mask, idx)
                    self.lbl_scores[k].setText(f"{score:.2f}")
            except ImportError:
                pass

            self.show_image(t1_img, self.views["input"])
            self.show_contours(t1_img, gt_mask, self.views["gt"])
            self.show_contours(t1_img, pred_mask, self.views["pred"])

    def show_image(self, img, label):
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def show_contours(self, img, mask, label):
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 顏色定義：1(MN)=黃色, 2(FT)=藍色, 3(CT)=紅色 (BGR格式)
        colors = {1: (0, 255, 255), 2: (255, 0, 0), 3: (0, 0, 255)}
        
        for cls_idx, color in colors.items():
            binary = np.zeros_like(mask, dtype=np.uint8)
            binary[mask == cls_idx] = 255
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bgr_img, cnts, -1, color, 2)

        h, w, ch = bgr_img.shape
        qimg = QImage(bgr_img.data.tobytes(), w, h, ch*w, QImage.Format.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CTSInspector()
    win.show()
    sys.exit(app.exec())