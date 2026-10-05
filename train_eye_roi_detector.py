import os, glob, json, math, re
import numpy as np
from PIL import Image

def train_eye_roi_model():
    print("=" * 60)
    print("  TRAINING AI OCULAR ROI & EYE REGION DETECTOR")
    print("  (Auto-Detects Left and Right Eye Regions Automatically)")
    print("=" * 60)

    gt_path = 'scratch/cornea_ground_truth.json'
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing {gt_path}")
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)

    print(f"Loaded ground truth for {len(ground_truth)} clinical subjects.")

    scale_y = np.arange(29, 209)
    t_max, t_min = 37.0, 27.0
    scale_temps = t_max - (scale_y - 29) / (208 - 29) * (t_max - t_min)

    # KDTree / Fast distance mapping using scikit-learn
    from sklearn.neighbors import NearestNeighbors
    
    def extract_eye_features(arr):
        # arr shape (240, 320, 3)
        h, w, _ = arr.shape
        sc = arr[scale_y, 308, :].astype(float)
        
        # Build 1-NN index for scale bar (180 colors)
        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(sc)
        
        # Downsample to 60x80
        down = arr[::4, ::4, :].reshape(-1, 3).astype(float)
        _, indices = nn.kneighbors(down)
        t_grid = scale_temps[indices.flatten()].reshape(60, 80)
        
        # Ocular zone: rows 20..46 (y: 80..184), cols 6..74 (x: 24..296)
        ocular_zone = t_grid[20:46, 6:74] # (26, 68)
        
        row_proj = np.mean(ocular_zone, axis=1) # 26 values
        col_proj = np.mean(ocular_zone, axis=0) # 68 values
        
        # Eye region candidate features
        # Right Eye (subject's right eye, image left side: cols 6..38)
        re_zone = t_grid[20:46, 6:38]
        re_min_idx = np.unravel_index(np.argmin(re_zone), re_zone.shape)
        re_est_x = (6 + re_min_idx[1]) * 4
        re_est_y = (20 + re_min_idx[0]) * 4
        
        # Left Eye (subject's left eye, image right side: cols 42..74)
        le_zone = t_grid[20:46, 42:74]
        le_min_idx = np.unravel_index(np.argmin(le_zone), le_zone.shape)
        le_est_x = (42 + le_min_idx[1]) * 4
        le_est_y = (20 + le_min_idx[0]) * 4
        
        # Medial canthus thermal landmarks (warmest ocular peaks)
        re_medial = np.max(t_grid[20:46, 30:40])
        le_medial = np.max(t_grid[20:46, 40:50])
        
        re_stats = [np.mean(re_zone), np.std(re_zone), np.min(re_zone), np.max(re_zone)]
        le_stats = [np.mean(le_zone), np.std(le_zone), np.min(le_zone), np.max(le_zone)]
        
        feats = np.concatenate([
            row_proj,
            col_proj,
            re_stats,
            le_stats,
            [re_est_x, re_est_y, le_est_x, le_est_y, re_medial, le_medial]
        ])
        return feats

    X = []
    Y_boxes = [] # [re_x1, re_y1, re_x2, re_y2, le_x1, le_y1, le_x2, le_y2]
    Y_centers = [] # [re_cx, re_cy, le_cx, le_cy]
    
    print("Extracting features across all subjects and time series...")
    eye_box_w = 40
    eye_box_h = 28
    
    count = 0
    for sno_str, gt in ground_truth.items():
        sno = int(sno_str)
        files = glob.glob(rf'samples\{sno} - *S.jpg')
        if not files:
            files = glob.glob(rf'samples\subject_{sno}_*S.jpg')
            
        rcx, rcy = gt['re']['cx'], gt['re']['cy']
        lcx, lcy = gt['le']['cx'], gt['le']['cy']
        
        box_targets = [
            max(0, rcx - eye_box_w), max(0, rcy - eye_box_h), min(320, rcx + eye_box_w), min(240, rcy + eye_box_h),
            max(0, lcx - eye_box_w), max(0, lcy - eye_box_h), min(320, lcx + eye_box_w), min(240, lcy + eye_box_h)
        ]
        center_targets = [rcx, rcy, lcx, lcy]
        
        for f in files:
            if not os.path.exists(f): continue
            try:
                im = Image.open(f)
                arr = np.array(im)
                if arr.shape != (240, 320, 3): continue
                fvec = extract_eye_features(arr)
                X.append(fvec)
                Y_boxes.append(box_targets)
                Y_centers.append(center_targets)
                count += 1
            except Exception:
                pass

    X = np.array(X, dtype=np.float32)
    Y_boxes = np.array(Y_boxes, dtype=np.float32)
    Y_centers = np.array(Y_centers, dtype=np.float32)
    
    print(f"Collected {count} image samples. Feature vector size: {X.shape[1]}")

    # Standardize
    mean_x = np.mean(X, axis=0)
    std_x = np.std(X, axis=0)
    std_x[std_x < 1e-6] = 1.0
    X_norm = (X - mean_x) / std_x

    # Joint multi-target regression for Eye Regions & Cornea Centers
    # Targets: [re_cx, re_cy, le_cx, le_cy, re_x1, re_y1, re_x2, re_y2, le_x1, le_y1, le_x2, le_y2] (12 outputs)
    Y_all = np.hstack([Y_centers, Y_boxes])
    
    X_design = np.hstack([X_norm, np.ones((X_norm.shape[0], 1), dtype=np.float32)])
    alpha = 8.0
    I = np.eye(X_design.shape[1], dtype=np.float32)
    I[-1, -1] = 0.0
    
    W = np.linalg.solve(X_design.T @ X_design + alpha * I, X_design.T @ Y_all)
    preds = X_design @ W
    
    center_errors = np.abs(preds[:, :4] - Y_centers)
    box_errors = np.abs(preds[:, 4:] - Y_boxes)
    
    # Calculate IoU for Right and Left eye boxes
    def calc_iou(b1, b2):
        # b is [x1, y1, x2, y2]
        ix1 = np.maximum(b1[:, 0], b2[:, 0])
        iy1 = np.maximum(b1[:, 1], b2[:, 1])
        ix2 = np.minimum(b1[:, 2], b2[:, 2])
        iy2 = np.minimum(b1[:, 3], b2[:, 3])
        iw = np.maximum(0, ix2 - ix1)
        ih = np.maximum(0, iy2 - iy1)
        inter = iw * ih
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = area1 + area2 - inter
        return np.mean(inter / np.maximum(1e-6, union))

    re_iou = calc_iou(preds[:, 4:8], Y_boxes[:, :4])
    le_iou = calc_iou(preds[:, 8:12], Y_boxes[:, 4:8])
    
    print("\n--- Model Evaluation ---")
    print(f"Right Eye Bounding Box Mean IoU: {re_iou * 100:.1f}%")
    print(f"Left Eye Bounding Box Mean IoU:  {le_iou * 100:.1f}%")
    print(f"Right Eye Center MAE: {np.mean(center_errors[:, 0]):.1f} px (x), {np.mean(center_errors[:, 1]):.1f} px (y)")
    print(f"Left Eye Center MAE:  {np.mean(center_errors[:, 2]):.1f} px (x), {np.mean(center_errors[:, 3]):.1f} px (y)")

    model_export = {
        'version': '2.0.0',
        'model_name': 'AI_Ocular_ROI_Eye_Region_Detector',
        'feature_dim': int(X.shape[1]),
        'num_targets': 12,
        'target_names': [
            're_cx', 're_cy', 'le_cx', 'le_cy',
            're_x1', 're_y1', 're_x2', 're_y2',
            'le_x1', 'le_y1', 'le_x2', 'le_y2'
        ],
        'mean_x': mean_x.tolist(),
        'std_x': std_x.tolist(),
        'weights': W[:-1, :].tolist(),
        'bias': W[-1, :].tolist(),
        'eye_box_dimensions': {'width': eye_box_w * 2, 'height': eye_box_h * 2},
        'cornea_radii': {'rx': 22, 'ry': 17}
    }

    with open('cornea_detector_weights.json', 'w') as f:
        json.dump(model_export, f, indent=2)

    with open('backend/models/cornea_detector_weights.json', 'w') as f:
        json.dump(model_export, f, indent=2)

    print("\nSaved trained eye ROI weights to cornea_detector_weights.json")
    return model_export

if __name__ == '__main__':
    train_eye_roi_model()
