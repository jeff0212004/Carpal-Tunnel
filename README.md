
# 醫學影像分割系統 - 腕隧道症候群 (CTS)

### Medical AI Segmentation System for Carpal Tunnel Syndrome

這是一個基於深度學習的自動化醫學影像分析系統，旨在從 T1 與 T2 MRI 序列中精確分割出正中神經 (MN)、屈肌腱 (FT) 及腕隧道 (CT) 。本系統整合了從數據預處理、模型訓練、多模型選拔到 GUI 互動式分析的完整開發流程 。

## 🌟 核心亮點


**雙序列輸入架構**：模型同時接收並融合 T1 與 T2 序列特徵，強化解剖組織的辨識度 。


 
**CLAHE 影像增強**：在資料讀取階段自動套用 CLAHE 技術，使神經與腱鞘的邊界更清晰 。



**U-Net++ 醫療級模型**：採用 `smp.UnetPlusPlus` 搭配 `efficientnet-b3` 骨幹網路，適合處理複雜的組織分割任務 。



**超級模型選拔機制**：創新的「Super Model」機制，能針對不同病例（Case）自動從 5-Fold 權重中挑選表現最優異的模型進行推論 。



**自定義 Dice 邏輯**：針對 CT 類別開發特殊的聯集處理（Union Logic），符合臨床上 CT 包含內部組織的解剖特徵 。



---

## 🛠️ 專案結構說明

本專案將核心邏輯封裝於 `src` 資料夾中，確保開發與展示的模組化 ：


**`src/config.py`**：全域設定檔，包含路徑、GPU 設備、類別權重及 1e-4 的學習率設定 。



**`src/model_arch.py`**：定義 U-Net++ 網路結構與影像通道輸入 。



**`src/engine.py`**：處理訓練循環與驗證流程，支援 `ReduceLROnPlateau` 學習率調整 。



**`src/metrics.py`**：計算 MN、FT、CT 的 Dice Score 分數 。



**`dataset_cts_v5.py`**：資料集載入器，包含三色標籤合成與 CLAHE 預處理 。



**`main_train.py`**：執行 5-Fold 交叉驗證訓練與早停機制（Early Stopping） 。



**`create_super_model.py`**：自動化評估所有病例，產出「超級模型包」。



**`app_gui.py`**：基於 PyQt6 的圖形化分析工具，支援即時分割與 Dice 分數顯示 。



---

## 🚀 快速開始指南

### 1. 資料路徑設定

請至 `src/config.py` 修改資料目錄位址 ：

```python
DATA_ROOT = r"C:\AI\carpalTunnel"

```

### 2. 啟動訓練

執行 5-Fold 交叉驗證訓練，模型將自動儲存於 `experiments_output` ：

```bash
python main_train.py

```

### 3. 生成超級模型 (展示專用)

訓練完成後，執行此腳本來打包各個病例的最佳權重 ：

```bash
python create_super_model.py

```

### 4. 開啟 GUI 展示介面

執行介面程式進行分析 ：

```bash
python app_gui.py

```

---

## 📊 分割類別定義

系統支援四種標籤 (Class IDs) ：

| ID | 名稱 | 解剖描述 | GUI 顯示顏色 |
| --- | --- | --- | --- |
| 0 | Background | 背景區域 | 無 |
| 1 | MN | 正中神經 (Median Nerve) | **黃色** |
| 2 | FT | 屈肌腱 (Flexor Tendons) | **藍色** |
| 3 | CT | 腕隧道 (Carpal Tunnel) | **紅色** |

---

## 🖥️ 展示介面功能


**自動載入機制**：支援讀取單一 `.pth` 或 `.pth` 超級模型包 。



**即時效能監控**：切換影像切片時，系統會顯示該張影像各類別的 Dice 分數 。



**輪廓對比顯示**：左側顯示 T1 原始影像，中間為 Ground Truth，右側為 AI 預測結果 。


