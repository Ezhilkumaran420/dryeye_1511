import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation attention block.
    Adaptively recalibrates channel-wise feature responses by modelling
    inter-channel dependencies.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """
    Standard ResNet Residual Block with skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Handles class imbalance by down-weighting well-classified examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.35, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets):
        probs = probs.clamp(1e-7, 1.0 - 1e-7)
        bce = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


class ThermalEyeCNN(nn.Module):
    """
    Deep Multi-Modal Thermal Eye CNN (ResNet-18-like):
      - Branch 1: Deep ResNet backbone with 2 blocks/stage and SE attention
                   on the final stage, processing 224x224x3 thermal eye images.
      - Branch 2: Deep Hotspot & Asymmetry MLP branch processing calibrated
                   ocular temperature features (18-dim).
      - Fusion: Joint dense projection with dropout and sigmoid activation.
    """
    def __init__(self, hotspot_feat_dim=18):
        super().__init__()
        # Image Feature Backbone (ResNet-18-like with SE attention)
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 2 blocks per stage (ResNet-18 pattern)
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=1),
            ResidualBlock(64, 64)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
            SEBlock(512, reduction=16)  # Squeeze-and-Excitation attention
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Deeper Thermal Hotspot & Asymmetry Branch (18 -> 64 -> 64 -> 48)
        self.hotspot_mlp = nn.Sequential(
            nn.Linear(hotspot_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 48),
            nn.ReLU(inplace=True)
        )

        # Multi-modal fusion classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.40),
            nn.Linear(512 + 48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        )

    def forward(self, img, hotspot_feats):
        # 1. Image branch
        x = self.prep(img)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        img_feats = self.pool(x).view(x.size(0), -1)  # (B, 512)

        # 2. Hotspot branch
        hs_feats = self.hotspot_mlp(hotspot_feats)     # (B, 48)

        # 3. Concatenation and classification
        fused = torch.cat([img_feats, hs_feats], dim=1)
        logits = self.classifier(fused)
        probs = torch.sigmoid(logits)
        return probs.squeeze(-1)


def augment_image_array(arr_224):
    """
    Applies real-time data augmentations specified in requirements:
      - Random rotation (±15°)
      - Horizontal flip (p=0.5)
      - Zoom (0.9 to 1.1)
      - Brightness adjustment (±10%)
      - Per-channel color jitter (±5%)
    """
    im = Image.fromarray((arr_224 * 255.0).astype(np.uint8))

    # 1. Random rotation ±15°
    angle = random.uniform(-15.0, 15.0)
    im = im.rotate(angle, resample=Image.Resampling.BILINEAR)

    # 2. Random horizontal flip
    if random.random() < 0.5:
        im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # 3. Random zoom (0.9 to 1.1)
    scale = random.uniform(0.90, 1.10)
    w, h = im.size
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    im_scaled = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
    if scale > 1.0:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        im = im_scaled.crop((left, top, left + w, top + h))
    else:
        new_canvas = Image.new('RGB', (w, h), (0, 0, 0))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        new_canvas.paste(im_scaled, (left, top))
        im = new_canvas

    # 4. Brightness adjustment ±10%
    b_factor = random.uniform(0.90, 1.10)
    arr = np.array(im, dtype=np.float32) / 255.0
    arr = np.clip(arr * b_factor, 0.0, 1.0)

    # 5. Per-channel color jitter ±5%
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] * random.uniform(0.95, 1.05), 0.0, 1.0)

    return arr


def mixup_batch(imgs, hotspot_feats, labels, alpha=0.2):
    """
    Applies Mixup augmentation: blends pairs of samples with random lambda
    drawn from Beta(alpha, alpha). Helps regularize and smooth decision boundary.
    """
    if alpha <= 0:
        return imgs, hotspot_feats, labels

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 so dominant label is preserved
    batch_size = imgs.size(0)
    index = torch.randperm(batch_size, device=imgs.device)

    mixed_imgs = lam * imgs + (1 - lam) * imgs[index]
    mixed_hs = lam * hotspot_feats + (1 - lam) * hotspot_feats[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_imgs, mixed_hs, mixed_labels


class ThermalModelTrainer:
    """
    Manages training, evaluation, early stopping, and inference for ThermalEyeCNN.
    Improvements over baseline:
      - Focal Loss for class imbalance handling
      - CosineAnnealingWarmRestarts LR scheduler
      - Mixup augmentation (50% of batches)
      - Gradient clipping (max_norm=1.0)
      - Patience-based early stopping (patience=10)
      - Evaluation every 2 epochs
      - AdamW optimizer with stronger weight decay
    """
    def __init__(self, device='cpu', lr=0.0002, epochs=50, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = ThermalEyeCNN(hotspot_feat_dim=18).to(self.device)
        self.scaler_mean = None
        self.scaler_std = None
        self.image_cache = {}

    def fit_scalers(self, train_samples):
        feats = np.array([s['feature_vector'] for s in train_samples], dtype=np.float32)
        self.scaler_mean = np.mean(feats, axis=0)
        self.scaler_std = np.std(feats, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0

    def transform_hotspot_feats(self, feats):
        feats_arr = np.array(feats, dtype=np.float32)
        if feats_arr.ndim == 1:
            feats_arr = feats_arr.reshape(1, -1)
        if self.scaler_mean is not None:
            feats_arr = (feats_arr - self.scaler_mean) / self.scaler_std
        return torch.tensor(feats_arr, dtype=torch.float32, device=self.device)

    def load_image_tensor(self, img_path, augment=False):
        if img_path not in self.image_cache:
            im = Image.open(img_path).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
            self.image_cache[img_path] = np.array(im, dtype=np.float32) / 255.0
        arr = self.image_cache[img_path]
        if augment:
            arr = augment_image_array(arr)
        # (224, 224, 3) -> (3, 224, 224)
        tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)
        return tensor

    def evaluate(self, samples):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i:i + self.batch_size]
                imgs = torch.stack([self.load_image_tensor(s['path'], augment=False) for s in batch]).to(self.device)
                hs_feats = np.array([s['feature_vector'] for s in batch], dtype=np.float32)
                hs_tensor = self.transform_hotspot_feats(hs_feats)
                probs = self.model(imgs, hs_tensor).cpu().numpy().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                all_preds.extend(probs)
                all_labels.extend([s['label'] for s in batch])

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        bin_preds = (preds_arr >= 0.50).astype(int)
        acc = float(np.mean(bin_preds == labels_arr))
        return acc, preds_arr, labels_arr

    def train(self, train_samples, val_samples):
        print(f"Beginning Thermal Image Model training on {self.device}...")
        self.fit_scalers(train_samples)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        criterion = FocalLoss(alpha=0.35, gamma=2.0)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        best_val_acc = 0.0
        best_state = None
        patience = 10
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            random.shuffle(train_samples)
            epoch_loss = 0.0
            batches = 0

            for i in range(0, len(train_samples), self.batch_size):
                batch = train_samples[i:i + self.batch_size]
                if len(batch) < 2:
                    continue  # Skip tiny batches (BatchNorm needs >= 2)

                imgs = torch.stack([self.load_image_tensor(s['path'], augment=True) for s in batch]).to(self.device)
                hs_feats = np.array([s['feature_vector'] for s in batch], dtype=np.float32)
                hs_tensor = self.transform_hotspot_feats(hs_feats)
                labels = torch.tensor([s['label'] for s in batch], dtype=torch.float32, device=self.device)

                # Mixup augmentation (50% of batches)
                if random.random() < 0.5:
                    imgs, hs_tensor, labels = mixup_batch(imgs, hs_tensor, labels, alpha=0.2)

                optimizer.zero_grad()
                probs = self.model(imgs, hs_tensor)
                loss = criterion(probs, labels)
                loss.backward()

                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(1, batches)
            current_lr = optimizer.param_groups[0]['lr']

            # Evaluate every 2 epochs or on first/last epoch
            if epoch % 2 == 0 or epoch == 1 or epoch == self.epochs:
                val_acc, _, _ = self.evaluate(val_samples)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                print(f"Epoch {epoch:3d}/{self.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} "
                      f"| Val Acc: {val_acc * 100:.2f}% (Best: {best_val_acc * 100:.2f}%) "
                      f"| Patience: {patience_counter}/{patience}")

                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} eval cycles).")
                    break

                if best_val_acc >= 0.99:
                    print(f"Reached optimal convergence ({best_val_acc * 100:.1f}%) at epoch {epoch}!")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"Thermal Model training complete! Optimal Val Accuracy: {best_val_acc * 100:.2f}%")
        return best_val_acc

    def predict_image(self, arr_224, feature_vector):
        """
        Inference on single image:
          - arr_224: (224, 224, 3) normalized [0, 1]
          - feature_vector: list of 18 hotspot metrics
        Returns float probability of Dry Eye (0.0 to 1.0)
        """
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(arr_224.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)
            hs_tensor = self.transform_hotspot_feats(feature_vector)
            prob = self.model(tensor, hs_tensor).item()
            return float(prob)

    def save_checkpoint(self, path=r"backend\models\thermal_cnn_model.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler_mean.tolist() if self.scaler_mean is not None else None,
            'scaler_std': self.scaler_std.tolist() if self.scaler_std is not None else None,
        }, path)
        print(f"Saved thermal CNN checkpoint to {path}")

    def load_checkpoint(self, path=r"backend\models\thermal_cnn_model.pt"):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        if checkpoint.get('scaler_mean') is not None:
            self.scaler_mean = np.array(checkpoint['scaler_mean'], dtype=np.float32)
            self.scaler_std = np.array(checkpoint['scaler_std'], dtype=np.float32)
        print(f"Loaded thermal CNN checkpoint from {path}")
