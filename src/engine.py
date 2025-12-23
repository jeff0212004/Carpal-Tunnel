import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from .metrics import calculate_dice_score
from .config import Config

class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.DEVICE
        self.weights = torch.tensor(Config.CLASS_WEIGHTS).to(self.device)

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0
        
        loop = tqdm(self.train_loader, desc=f"Train Ep {epoch_idx}", leave=False)
        for imgs, masks in loop:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device).long()

            # Forward
            outputs = self.model(imgs)
            loss = F.cross_entropy(outputs, masks, weight=self.weights)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for imgs, masks in self.val_loader:
                imgs = imgs.to(self.device)
                masks = masks.numpy()[0] # Batch size 1 for validation

                outputs = self.model(imgs)
                pred_mask = torch.argmax(outputs, dim=1).cpu().numpy()[0]

                # 計算關鍵指標 (MN 和 CT)
                d1 = calculate_dice_score(pred_mask, masks, 1)
                d3 = calculate_dice_score(pred_mask, masks, 3)
                scores.append((d1 + d3) / 2.0)
                
        return np.mean(scores)