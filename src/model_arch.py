import torch.nn as nn
import segmentation_models_pytorch as smp

class SegmentationNet(nn.Module):
    def __init__(self, n_classes=4, encoder="efficientnet-b3", weights="imagenet"):
        """
        初始化分割網路
        Args:
            n_classes: 分割類別數 (背景 + 3個組織)
            encoder: 骨幹網路選擇 (使用 EfficientNet 获取更好特徵)
            weights: 預訓練權重 (Transfer Learning)
        """
        super().__init__()
        
        # 使用 U-Net++ 架構，這在醫療影像分割中表現優異
        self.arc = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=2,  # 輸入包含 T1 與 T2 兩張影像
            classes=n_classes,
        )

    def forward(self, x):
        return self.arc(x)