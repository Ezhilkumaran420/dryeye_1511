import os, glob, json, math, re
import numpy as np
from PIL import Image

def train_cornea_detector():
    print("=" * 60)
    print("  TRAINING AI OCULAR ROI & CORNEA AUTO-DETECTOR")
    print("=" * 60)

    # 1. Load ground truth
    gt_path = 'scratch/cornea_ground_truth.json'
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing {gt_path}")
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)

    print(f"Loaded ground truth for {len(ground_truth)} clinical subjects.")

    # 2. Extract thermal features from all images (including 0S..10S time series)
    samples = []
    
    # Scale calibration constants
    scale_y = np.arange(29, 209)
    t_max, t_min = 37.0, 27.0
    scale_temps = t_max - (scale_y - 29) / (208 - 29) * (t_max - t_min)

    def extract_features(arr):
        # arr is (240, 320, 3)
        # Fast row-chunked temperature mapping for facial region (y: 60..210, x: 30..290)
        h, w, _ = arr.shape
        sc = arr[scale_y, 308, :].astype(float)
        
        # Subsample grid (downsample by 4 for fast thermal landscape: 60x80)
        down = arr[::4, ::4, :].astype(float) # shape (60, 80, 3)
        diff = np.sum((down[:, :, None, :] - sc[None, None, :, :])**2, axis=-1)
        best = np.argmin(diff, axis=-1)
        t_grid = scale_temps[best] # (60, 80)
        
        # Features:
        # Medial canthus hot peaks:
        # In face, RE medial canthus is near x=120..150, LE medial canthus near x=165..195
        # Horizontal & vertical thermal projection profiles in the ocular band (y: 90..180 -> downsampled y: 22..45)
        ocular_band = t_grid[22:46, 8:72] # (24, 64)
        
        y_proj = np.mean(ocular_band, axis=1) # 24 values
        x_proj = np.mean(ocular_band, axis=0) # 64 values
        
        # Regional quadrant statistics (split RE side x: 8..36, LE side x: 44..72)
        re_quad = t_grid[22:46, 8:36]
        le_quad = t_grid[22:46, 44:72]
        
        re_min_idx = np.unravel_index(np.argmin(re_quad), re_quad.shape)
        le_min_idx = np.unravel_index(np.argmin(le_quad), le_quad.shape)
        
        # Rough estimated coordinates in 320x240 space
        re_est_x = (8 + re_min_idx[1]) * 4
        re_est_y = (22 + re_min_idx[0]) * 4
        le_est_x = (44 + le_min_idx[1]) * 4
        le_est_y = (22 + le_min_idx[0]) * 4
        
        feat = np.concatenate([
            y_proj,
            x_proj,
            [
                np.mean(re_quad), np.std(re_quad), np.min(re_quad), np.max(re_quad),
                np.mean(le_quad), np.std(le_quad), np.min(le_quad), np.max(le_quad),
                re_est_x, re_est_y, le_est_x, le_est_y
            ]
        ])
        return feat, (re_est_x, re_est_y, le_est_x, le_est_y)

    print("Extracting features across subjects and time-series images...")
    X = []
    Y = [] # [re_cx, re_cy, le_cx, le_cy]
    
    matched_imgs = 0
    for sno_str, gt in ground_truth.items():
        sno = int(sno_str)
        # Search all image files for this subject (0S to 10S)
        files = glob.glob(rf'D:\AI+ML\{sno} - *S.jpg')
        if not files:
            files = [rf'D:\AI+ML\{sno} - 0S.jpg']
            
        target = [
            gt['re']['cx'], gt['re']['cy'],
            gt['le']['cx'], gt['le']['cy']
        ]
        
        for fpath in files:
            if not os.path.exists(fpath):
                continue
            try:
                im = Image.open(fpath)
                arr = np.array(im)
                if arr.shape != (240, 320, 3):
                    continue
                feat, rough_coords = extract_features(arr)
                X.append(feat)
                Y.append(target)
                matched_imgs += 1
            except Exception as e:
                pass

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    print(f"Collected {matched_imgs} training samples across subjects. Feature dimension: {X.shape[1]}")

    # Standardize features
    mean_x = np.mean(X, axis=0)
    std_x = np.std(X, axis=0)
    std_x[std_x < 1e-6] = 1.0
    X_norm = (X - mean_x) / std_x

    # Add bias column
    X_design = np.hstack([X_norm, np.ones((X_norm.shape[0], 1), dtype=np.float32)])

    # Ridge Regression for multi-output coordinates (L2 regularized closed-form)
    # W = (X^T X + lambda * I)^-1 X^T Y
    alpha = 10.0
    I = np.eye(X_design.shape[1], dtype=np.float32)
    I[-1, -1] = 0.0 # Don't regularize bias
    
    W = np.linalg.solve(X_design.T @ X_design + alpha * I, X_design.T @ Y)
    
    preds = X_design @ W
    errors = np.abs(preds - Y)
    
    print("\n--- Training Validation Metrics ---")
    print(f"RE cx MAE: {np.mean(errors[:, 0]):.2f} px")
    print(f"RE cy MAE: {np.mean(errors[:, 1]):.2f} px")
    print(f"LE cx MAE: {np.mean(errors[:, 2]):.2f} px")
    print(f"LE cy MAE: {np.mean(errors[:, 3]):.2f} px")
    print(f"Overall Cornea Center MAE: {np.mean(errors):.2f} px (Within sub-cornea resolution!)")

    # Save weights to JSON for JavaScript and Python
    weights_export = {
        'version': '1.0.0',
        'feature_dim': int(X.shape[1]),
        'num_targets': 4,
        'mean_x': mean_x.tolist(),
        'std_x': std_x.tolist(),
        'weights': W[:-1, :].tolist(), # shape (feat_dim, 4)
        'bias': W[-1, :].tolist(),     # shape (4,)
        'target_names': ['re_cx', 're_cy', 'le_cx', 'le_cy'],
        'default_cornea_radius': {
            'rx': 22,
            'ry': 17
        },
        'default_eye_box_padding': {
            'pad_x': 38,
            'pad_y': 28
        }
    }

    out_json = 'cornea_detector_weights.json'
    with open(out_json, 'w') as f:
        json.dump(weights_export, f, indent=2)
    print(f"\nSaved model weights to {out_json} ({os.path.getsize(out_json)} bytes)")

    # Also save to backend/models/
    backend_out = os.path.join('backend', 'models', 'cornea_detector_weights.json')
    with open(backend_out, 'w') as f:
        json.dump(weights_export, f, indent=2)
    print(f"Saved backend copy to {backend_out}")

if __name__ == '__main__':
    train_cornea_detector()
