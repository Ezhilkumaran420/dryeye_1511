import openpyxl
import numpy as np

wb_eye = openpyxl.load_workbook(r"C:\Users\ezhil\Downloads\Eye classification.xlsx", data_only=True)

samples = []
for eye in ['RE', 'LE']:
    ws = wb_eye[eye]
    rows = list(ws.iter_rows(values_only=True))[1:]
    for r in rows:
        lbl = str(r[-1]).strip().lower()
        y = 1 if 'dry' in lbl else (0 if 'normal' in lbl else None)
        if y is None: continue
        
        osdi = float(r[1] or 0)
        cr10 = float(r[4 if eye=='RE' else 2] or 0)
        cr7  = float(r[5 if eye=='RE' else 3] or 0)
        nc   = float(r[6 if eye=='RE' else 4] or 0)
        tc   = float(r[7 if eye=='RE' else 5] or 0)
        cc   = float(r[8 if eye=='RE' else 6] or 0)
        nl   = float(r[9 if eye=='RE' else 7] or 0)
        tl   = float(r[10 if eye=='RE' else 8] or 0)
        t0   = float(r[11 if eye=='RE' else 9] or 0)
        t10  = float(r[12 if eye=='RE' else 10] or 0)
        most = float(r[13 if eye=='RE' else 11] or 0)
        
        samples.append({
            'eye': eye,
            'y': y,
            'features_full': [osdi, cr10, cr7, nc, tc, cc, nl, tl, t0, t10, most],
            'features_thermal_only': [cr10, cr7, nc, tc, cc, nl, tl, t0, t10, most]
        })

print(f"Total samples: {len(samples)}, Positives: {sum(s['y'] for s in samples)}, Negatives: {len(samples) - sum(s['y'] for s in samples)}")

def train_eval(feat_key):
    X = np.array([s[feat_key] for s in samples])
    y = np.array([s['y'] for s in samples])
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std
    
    N, D = X_norm.shape
    w = np.zeros(D)
    b = 0.0
    
    pos_weight = (len(y) - sum(y)) / sum(y) # Balanced weighting: ~ 110 / 46 = 2.39
    
    lr = 0.08
    epochs = 2500
    reg = 0.01
    
    for _ in range(epochs):
        z = np.dot(X_norm, w) + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
        
        # Weighted error
        err = (p - y)
        weights = np.where(y == 1, pos_weight, 1.0)
        weighted_err = err * weights
        
        grad_w = np.dot(X_norm.T, weighted_err) / N + reg * w
        grad_b = np.mean(weighted_err)
        
        w -= lr * grad_w
        b -= lr * grad_b
        
    z_final = np.dot(X_norm, w) + b
    probs = 1.0 / (1.0 + np.exp(-np.clip(z_final, -15.0, 15.0)))
    
    # Best threshold search
    best_acc, best_thresh = 0, 0.5
    for thresh in np.linspace(0.2, 0.8, 61):
        preds = (probs >= thresh).astype(int)
        acc = np.mean(preds == y)
        tp = np.sum((preds == 1) & (y == 1))
        fn = np.sum((preds == 0) & (y == 1))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        tn = np.sum((preds == 0) & (y == 0))
        fp = np.sum((preds == 1) & (y == 0))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        if acc > best_acc and sens >= 0.80 and spec >= 0.80:
            best_acc = acc
            best_thresh = thresh
            
    preds = (probs >= best_thresh).astype(int)
    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = np.mean(preds == y)
    
    print(f"\nModel: {feat_key}")
    print(f"  Accuracy:    {acc*100:.2f}% (Threshold={best_thresh:.2f})")
    print(f"  Sensitivity: {sens*100:.2f}% ({tp}/{tp+fn} dry detected)")
    print(f"  Specificity: {spec*100:.2f}% ({tn}/{tn+fp} normal confirmed)")
    print(f"  Confusion:   TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Weights:     {np.round(w, 3).tolist()}")
    print(f"  Bias:        {b:.3f}")
    return w, b, mean, std, best_thresh

train_eval('features_full')
train_eval('features_thermal_only')
