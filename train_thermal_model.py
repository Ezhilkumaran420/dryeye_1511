import os, re, glob, json, shutil
import openpyxl
from PIL import Image
import numpy as np

def extract_thermal_metrics(img_path):
    """
    Extracts calibrated ocular surface temperature metrics from an anterior segment FLIR thermal image.
    Uses scale bar color calibration and anatomical ocular ROI segmentation.
    Enhanced with additional derived features for improved classification.
    """
    if not os.path.exists(img_path):
        return None
        
    try:
        im = Image.open(img_path)
        arr = np.array(im)
        
        # Scale bar calibration (column 308, y: 29..208)
        scale_y = np.arange(29, 209)
        scale_colors = arr[scale_y, 308, :].astype(float) # shape (180, 3)
        t_max = 37.0
        t_min = 27.0
        scale_temps = t_max - (scale_y - 29) / (208 - 29) * (t_max - t_min)
        
        def process_eye_roi(crop):
            h, w, _ = crop.shape
            flat = crop.reshape(-1, 3).astype(float)
            diff = np.sum((flat[:, None, :] - scale_colors[None, :, :])**2, axis=-1)
            best = np.argmin(diff, axis=-1)
            temps = scale_temps[best].reshape(h, w)
            
            # Central cornea (larger center patch for more robust measurement)
            cy, cx = h // 2, w // 2
            r_y, r_x = 14, 18
            cc_patch = temps[max(0, cy-r_y):min(h, cy+r_y), max(0, cx-r_x):min(w, cx+r_x)]
            
            # Nasal cornea (inner side)
            nc_patch = temps[max(0, cy-12):min(h, cy+12), min(w-20, cx+12):min(w, cx+32)]
            # Temporal cornea (outer side)
            tc_patch = temps[max(0, cy-12):min(h, cy+12), max(0, cx-32):max(12, cx-12)]
            
            # Superior cornea (upper region)
            sc_patch = temps[max(0, cy-r_y-8):max(0, cy-4), max(0, cx-r_x):min(w, cx+r_x)]
            # Inferior cornea (lower region)
            ic_patch = temps[min(h, cy+4):min(h, cy+r_y+8), max(0, cx-r_x):min(w, cx+r_x)]
            
            t_cc = float(np.mean(cc_patch)) if cc_patch.size > 0 else float(np.mean(temps))
            t_nc = float(np.mean(nc_patch)) if nc_patch.size > 0 else t_cc + 0.5
            t_tc = float(np.mean(tc_patch)) if tc_patch.size > 0 else t_cc - 0.5
            t_sc = float(np.mean(sc_patch)) if sc_patch.size > 0 else t_cc
            t_ic = float(np.mean(ic_patch)) if ic_patch.size > 0 else t_cc
            t_mean = float(np.mean(temps))
            t_median = float(np.median(temps))
            t_min_val = float(np.percentile(temps, 5))
            t_max_val = float(np.percentile(temps, 95))
            t_p10 = float(np.percentile(temps, 10))
            t_p25 = float(np.percentile(temps, 25))
            t_p75 = float(np.percentile(temps, 75))
            t_std = float(np.std(temps))
            t_iqr = t_p75 - t_p25
            
            # Cold spots indicating localized tear film breakup
            cold_spots_ratio = float(np.mean(temps < 34.2))
            # Warm spots ratio (well-perfused areas)
            warm_spots_ratio = float(np.mean(temps > 35.5))
            
            # Skewness of temperature distribution (asymmetric cooling = dry eye sign)
            t_skew = float(np.mean(((temps - t_mean) / max(t_std, 0.01)) ** 3))
            # Kurtosis (peakedness of distribution)
            t_kurtosis = float(np.mean(((temps - t_mean) / max(t_std, 0.01)) ** 4) - 3.0)
            
            return {
                't_cc': round(t_cc, 2),
                't_nc': round(t_nc, 2),
                't_tc': round(t_tc, 2),
                't_sc': round(t_sc, 2),
                't_ic': round(t_ic, 2),
                't_mean': round(t_mean, 2),
                't_median': round(t_median, 2),
                't_min': round(t_min_val, 2),
                't_max': round(t_max_val, 2),
                't_p10': round(t_p10, 2),
                't_p25': round(t_p25, 2),
                't_p75': round(t_p75, 2),
                't_std': round(t_std, 2),
                't_iqr': round(t_iqr, 2),
                't_skew': round(t_skew, 3),
                't_kurtosis': round(t_kurtosis, 3),
                'cold_spots_ratio': round(cold_spots_ratio, 3),
                'warm_spots_ratio': round(warm_spots_ratio, 3)
            }
            
        re_metrics = process_eye_roi(arr[108:172, 48:142])
        le_metrics = process_eye_roi(arr[108:172, 173:267])
        
        return {
            're': re_metrics,
            'le': le_metrics
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def build_enhanced_features(osdi, cr, metrics):
    """
    Build an enhanced feature vector with derived clinical biomarkers.
    Returns a list of features that capture both raw measurements and
    clinically meaningful derived indicators of dry eye disease.
    """
    t_cc = metrics['t_cc']
    t_nc = metrics['t_nc']
    t_tc = metrics['t_tc']
    t_sc = metrics['t_sc']
    t_ic = metrics['t_ic']
    t_mean = metrics['t_mean']
    t_median = metrics['t_median']
    t_min = metrics['t_min']
    t_max = metrics['t_max']
    t_p10 = metrics['t_p10']
    t_p25 = metrics['t_p25']
    t_p75 = metrics['t_p75']
    t_std = metrics['t_std']
    t_iqr = metrics['t_iqr']
    t_skew = metrics['t_skew']
    t_kurtosis = metrics['t_kurtosis']
    cold_ratio = metrics['cold_spots_ratio']
    warm_ratio = metrics['warm_spots_ratio']
    
    # --- Derived clinical features ---
    # Nasal-Temporal gradient (asymmetry within the cornea)
    nt_gradient = t_nc - t_tc
    # Superior-Inferior gradient
    si_gradient = t_sc - t_ic
    # Central cornea vs mean surface differential (corneal cooling indicator)
    cc_mean_diff = t_cc - t_mean
    # Thermal range (wider = more heterogeneous surface = possible dry eye)
    thermal_range = t_max - t_min
    # Central cornea deviation from median
    cc_median_diff = t_cc - t_median
    # OSDI symptom severity indicator (binary-like)
    osdi_high = 1.0 if osdi > 20 else 0.0
    osdi_moderate = 1.0 if 12 < osdi <= 20 else 0.0
    # OSDI × thermal interaction (symptomatic + cold cornea = strong signal)
    osdi_cc_interaction = osdi * max(0, 35.0 - t_cc)
    osdi_cold_interaction = osdi * cold_ratio
    # Cooling rate features
    cr_magnitude = abs(cr)
    cr_negative = 1.0 if cr < -0.02 else 0.0
    # Cold/warm ratio (balance indicator)
    cold_warm_balance = cold_ratio - warm_ratio
    # Low central cornea temperature flag (< 34.5°C is clinically significant)
    low_cc_flag = max(0.0, 34.5 - t_cc)
    # Percentile spread features
    lower_tail_spread = t_p25 - t_min
    upper_tail_spread = t_max - t_p75
    
    features = [
        # Original base features (10)
        osdi,
        cr,
        t_nc,
        t_tc,
        t_cc,
        t_mean,
        t_min,
        t_max,
        t_std,
        cold_ratio,
        # New raw features (8)
        t_sc,
        t_ic,
        t_median,
        t_p10,
        t_iqr,
        t_skew,
        t_kurtosis,
        warm_ratio,
        # Derived clinical features (14)
        nt_gradient,
        si_gradient,
        cc_mean_diff,
        thermal_range,
        cc_median_diff,
        osdi_high,
        osdi_moderate,
        osdi_cc_interaction,
        osdi_cold_interaction,
        cr_magnitude,
        cr_negative,
        cold_warm_balance,
        low_cc_flag,
        lower_tail_spread,
        upper_tail_spread,
    ]
    
    return features


# ============== MLP Neural Network (pure NumPy) ==============

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

def mlp_forward(X, params):
    """Forward pass through 2-hidden-layer MLP."""
    W1, b1, W2, b2, W3, b3 = params
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    out = sigmoid(z3.flatten())
    return out, (z1, a1, z2, a2, z3)

def mlp_train(X, y, hidden1=32, hidden2=16, lr=0.01, epochs=3000, reg=0.001,
              class_weight_ratio=None, verbose=True):
    """
    Train a 2-hidden-layer MLP with class-weighted BCE loss.
    Uses He initialization and Adam optimizer.
    """
    N, D = X.shape
    np.random.seed(42)
    
    # He initialization
    W1 = np.random.randn(D, hidden1) * np.sqrt(2.0 / D)
    b1 = np.zeros(hidden1)
    W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
    b2 = np.zeros(hidden2)
    W3 = np.random.randn(hidden2, 1) * np.sqrt(2.0 / hidden2)
    b3 = np.zeros(1)
    
    params = [W1, b1, W2, b2, W3, b3]
    
    # Class weights for imbalanced data
    if class_weight_ratio is None:
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        w_pos = N / (2.0 * max(n_pos, 1))
        w_neg = N / (2.0 * max(n_neg, 1))
    else:
        w_neg, w_pos = 1.0, class_weight_ratio
    
    sample_weights = np.where(y == 1, w_pos, w_neg)
    
    # Adam optimizer state
    adam_m = [np.zeros_like(p) for p in params]
    adam_v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    best_loss = float('inf')
    patience_counter = 0
    best_params = [p.copy() for p in params]
    
    for epoch in range(epochs):
        # Forward
        W1, b1, W2, b2, W3, b3 = params
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = relu(z2)
        z3 = (a2 @ W3 + b3).flatten()
        probs = sigmoid(z3)
        
        # Weighted BCE loss
        probs_clip = np.clip(probs, 1e-7, 1 - 1e-7)
        loss = -np.mean(sample_weights * (y * np.log(probs_clip) + (1 - y) * np.log(1 - probs_clip)))
        # L2 regularization
        loss += reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        
        # Track best
        if loss < best_loss - 1e-6:
            best_loss = loss
            best_params = [p.copy() for p in params]
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter > 500:
            if verbose:
                print(f"  Early stopping at epoch {epoch}, loss={loss:.4f}")
            break
        
        # Backward
        dz3 = (sample_weights * (probs - y)).reshape(-1, 1) / N
        dW3 = a2.T @ dz3 + 2 * reg * W3
        db3 = np.sum(dz3, axis=0)
        
        da2 = dz3 @ W3.T
        dz2 = da2 * relu_deriv(z2)
        dW2 = a1.T @ dz2 + 2 * reg * W2
        db2 = np.sum(dz2, axis=0)
        
        da1 = dz2 @ W2.T
        dz1 = da1 * relu_deriv(z1)
        dW1 = X.T @ dz1 + 2 * reg * W1
        db1 = np.sum(dz1, axis=0)
        
        grads = [dW1, db1, dW2, db2, dW3, db3]
        
        # Adam update
        t = epoch + 1
        for i in range(len(params)):
            adam_m[i] = beta1 * adam_m[i] + (1 - beta1) * grads[i]
            adam_v[i] = beta2 * adam_v[i] + (1 - beta2) * grads[i]**2
            m_hat = adam_m[i] / (1 - beta1**t)
            v_hat = adam_v[i] / (1 - beta2**t)
            params[i] = params[i] - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        if verbose and epoch % 500 == 0:
            preds = (probs >= 0.5).astype(int)
            acc = np.mean(preds == y)
            print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc*100:.1f}%")
    
    return best_params


def find_optimal_threshold(probs, y):
    """Find the threshold that maximizes F1-score."""
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.20, 0.65, 0.01):
        preds = (probs >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return round(best_thresh, 2), best_f1


def smote_oversample(X, y, target_ratio=1.0, k=5, random_state=42):
    """
    Simple SMOTE-like oversampling of the minority class.
    Generates synthetic samples by interpolating between minority class neighbors.
    """
    rng = np.random.RandomState(random_state)
    
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]
    
    n_minority = len(minority_idx)
    n_majority = len(majority_idx)
    n_synthetic = int(n_majority * target_ratio) - n_minority
    
    if n_synthetic <= 0:
        return X, y
    
    X_min = X[minority_idx]
    
    # Find k nearest neighbors within minority class
    synthetic_samples = []
    for _ in range(n_synthetic):
        idx = rng.randint(0, n_minority)
        anchor = X_min[idx]
        
        # Compute distances to other minority samples
        dists = np.sum((X_min - anchor)**2, axis=1)
        dists[idx] = np.inf  # Exclude self
        nn_indices = np.argsort(dists)[:min(k, n_minority - 1)]
        
        # Pick a random neighbor and interpolate
        nn_idx = rng.choice(nn_indices)
        lam = rng.uniform(0, 1)
        synthetic = anchor + lam * (X_min[nn_idx] - anchor)
        synthetic_samples.append(synthetic)
    
    X_synth = np.array(synthetic_samples)
    y_synth = np.ones(n_synthetic)
    
    X_out = np.vstack([X, X_synth])
    y_out = np.concatenate([y, y_synth])
    
    # Shuffle
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def train_and_export_thermal_system():
    thermal_dir = r"D:\research (dry eyes ml)\AI+ML"
    excel_path = r"C:\Users\shamk\Downloads\Eye classification.xlsx"
    npd_path = r"C:\Users\shamk\Downloads\N & PD Eye.xlsx"
    
    print("Loading clinical datasets from Excel...")
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    
    # Extract clinical ground truth
    re_clinical = {}
    for r in list(wb['RE'].iter_rows(values_only=True))[1:]:
        sno = int(r[0])
        lbl = str(r[-1]).strip().lower()
        y = 0 if 'normal' in lbl else (1 if 'dry' in lbl else None)
        if y is not None:
            re_clinical[sno] = {
                'label': y,
                'label_str': 'Normal' if y == 0 else 'Possible Dry Eye',
                'osdi': float(r[1] or 0),
                'age': r[2],
                'gender': r[3],
                'cr10_excel': float(r[4] or 0),
                'nc_excel': float(r[6] or 0),
                'tc_excel': float(r[7] or 0),
                'cc_excel': float(r[8] or 0),
                't0_excel': float(r[11] or 0),
                't10_excel': float(r[12] or 0),
                'most_excel': float(r[13] or 0)
            }
            
    le_clinical = {}
    for r in list(wb['LE'].iter_rows(values_only=True))[1:]:
        sno = int(r[0])
        lbl = str(r[-1]).strip().lower()
        y = 0 if 'normal' in lbl else (1 if 'dry' in lbl else None)
        if y is not None:
            le_clinical[sno] = {
                'label': y,
                'label_str': 'Normal' if y == 0 else 'Possible Dry Eye',
                'osdi': float(r[1] or 0),
                'cr10_excel': float(r[2] or 0),
                'nc_excel': float(r[4] or 0),
                'tc_excel': float(r[5] or 0),
                'cc_excel': float(r[6] or 0),
                't0_excel': float(r[9] or 0),
                't10_excel': float(r[10] or 0),
                'most_excel': float(r[11] or 0)
            }
            
    print(f"Loaded ground truth: {len(re_clinical)} RE, {len(le_clinical)} LE subjects.")
    
    # Process thermal images in D:\AI+ML
    print("Processing thermal images from D:\\AI+ML...")
    subjects_dataset = []
    
    # Also prepare a web samples directory with representative thermal images
    web_samples_dir = "samples"
    os.makedirs(web_samples_dir, exist_ok=True)
    
    sample_gallery = []
    
    for sno in sorted(re_clinical.keys()):
        f0 = os.path.join(thermal_dir, f"{sno} - 0S.jpg")
        f10 = os.path.join(thermal_dir, f"{sno} - 10S.jpg")
        
        # Regex search if spacing differs
        if not os.path.exists(f0):
            cands = [f for f in os.listdir(thermal_dir) if re.match(rf"^{sno}\s*-\s*0s?\.jpg$", f, re.I)]
            if cands: f0 = os.path.join(thermal_dir, cands[0])
        if not os.path.exists(f10):
            cands = [f for f in os.listdir(thermal_dir) if re.match(rf"^{sno}\s*-\s*10s?\.jpg$", f, re.I)]
            if cands: f10 = os.path.join(thermal_dir, cands[0])
            
        if not os.path.exists(f0):
            continue
            
        metrics0 = extract_thermal_metrics(f0)
        metrics10 = extract_thermal_metrics(f10) if os.path.exists(f10) else None
        
        if not metrics0:
            continue
            
        # Calculate dynamic cooling rate from images
        re_cr = round((metrics10['re']['t_cc'] - metrics0['re']['t_cc']) / 10.0, 3) if metrics10 else -0.025
        le_cr = round((metrics10['le']['t_cc'] - metrics0['le']['t_cc']) / 10.0, 3) if metrics10 else -0.025
        
        # Right Eye entry with enhanced features
        re_data = re_clinical[sno]
        re_features = build_enhanced_features(re_data['osdi'], re_cr, metrics0['re'])
        subjects_dataset.append({
            'sno': sno,
            'eye': 'RE',
            'label': re_data['label'],
            'label_str': re_data['label_str'],
            'features': re_features,
            'base_metrics': metrics0['re']
        })
        
        # Left Eye entry with enhanced features
        le_data = le_clinical.get(sno, re_data)
        le_features = build_enhanced_features(le_data['osdi'], le_cr, metrics0['le'])
        subjects_dataset.append({
            'sno': sno,
            'eye': 'LE',
            'label': le_data['label'],
            'label_str': le_data['label_str'],
            'features': le_features,
            'base_metrics': metrics0['le']
        })
        
        # Copy curated sample gallery images for web presentation (Normal, Dry Eye, and Asymmetric)
        if sno in [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 18, 30, 60]:
            sample_fn = f"subject_{sno}_0S.jpg"
            target_sample = os.path.join(web_samples_dir, sample_fn)
            shutil.copy(f0, target_sample)
            
            # Determine category string
            re_is_dry = (re_data['label'] == 1)
            le_is_dry = (le_data['label'] == 1)
            if re_is_dry and le_is_dry:
                cat_desc = "Bilateral Dry Eye"
            elif not re_is_dry and not le_is_dry:
                cat_desc = "Healthy Normal"
            elif not re_is_dry and le_is_dry:
                cat_desc = "Asymmetric (RE Normal / LE Dry)"
            else:
                cat_desc = "Asymmetric (RE Dry / LE Normal)"

            sample_gallery.append({
                'id': f"Subject #{sno}",
                'filename': sample_fn,
                'path': f"samples/{sample_fn}",
                'category': cat_desc,
                're_diagnosis': re_data['label_str'],
                'le_diagnosis': le_data['label_str'],
                'osdi': re_data['osdi'],
                're_metrics': metrics0['re'],
                'le_metrics': metrics0['le'],
                're_cc': metrics0['re']['t_cc'],
                'le_cc': metrics0['le']['t_cc'],
                're_cr': re_cr,
                'le_cr': le_cr
            })
            
    print(f"Constructed combined dataset of {len(subjects_dataset)} eye samples.")
    
    # ============ Prepare Data ============
    X_raw = np.array([s['features'] for s in subjects_dataset])
    y_raw = np.array([s['label'] for s in subjects_dataset])
    
    n_pos = int(np.sum(y_raw == 1))
    n_neg = int(np.sum(y_raw == 0))
    print(f"Class distribution: {n_neg} Normal, {n_pos} Dry Eye (ratio: 1:{n_neg/max(n_pos,1):.1f})")
    
    # Normalize features
    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0)
    std[std == 0] = 1.0
    X_norm = (X_raw - mean) / std
    
    # SMOTE oversampling to balance classes
    print("\nApplying SMOTE oversampling to balance classes...")
    X_train, y_train = smote_oversample(X_norm, y_raw, target_ratio=0.8, k=5)
    n_pos_aug = int(np.sum(y_train == 1))
    n_neg_aug = int(np.sum(y_train == 0))
    print(f"After SMOTE: {n_neg_aug} Normal, {n_pos_aug} Dry Eye (N={len(y_train)})")
    
    # ============ Train MLP Neural Network ============
    N_feat = X_norm.shape[1]
    print(f"\nTraining MLP Neural Network ({N_feat} features -> 32 -> 16 -> 1)...")
    
    mlp_params = mlp_train(
        X_train, y_train,
        hidden1=32, hidden2=16,
        lr=0.005, epochs=4000, reg=0.002,
        verbose=True
    )
    
    # Evaluate on original (non-augmented) data
    mlp_probs, _ = mlp_forward(X_norm, mlp_params)
    
    # Find optimal threshold
    optimal_thresh, best_f1 = find_optimal_threshold(mlp_probs, y_raw)
    print(f"\nOptimal threshold: {optimal_thresh} (F1={best_f1:.3f})")
    
    mlp_preds = (mlp_probs >= optimal_thresh).astype(int)
    
    accuracy = float(np.mean(mlp_preds == y_raw))
    tp = int(np.sum((mlp_preds == 1) & (y_raw == 1)))
    tn = int(np.sum((mlp_preds == 0) & (y_raw == 0)))
    fp = int(np.sum((mlp_preds == 1) & (y_raw == 0)))
    fn = int(np.sum((mlp_preds == 0) & (y_raw == 1)))
    
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sens / (precision + sens) if (precision + sens) > 0 else 0.0
    
    N_total = len(y_raw)
    print(f"\n{'='*55}")
    print(f" Thermal Image MLP Model Performance (N={N_total})")
    print(f"{'='*55}")
    print(f"  Accuracy:    {accuracy * 100:.2f}%")
    print(f"  Sensitivity: {sens * 100:.2f}%")
    print(f"  Specificity: {spec * 100:.2f}%")
    print(f"  Precision:   {precision * 100:.2f}%")
    print(f"  F1 Score:    {f1 * 100:.2f}%")
    print(f"  Threshold:   {optimal_thresh}")
    print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"{'='*55}")
    
    # ============ Also train a Logistic Regression for ensemble fallback ============
    print("\nTraining Logistic Regression baseline for ensemble...")
    N_lr, D_lr = X_train.shape
    w_lr = np.zeros(D_lr)
    b_lr = 0.0
    lr_rate = 0.15
    lr_epochs = 2000
    lr_reg = 0.03
    
    # Class-weighted logistic regression
    sample_w_lr = np.where(y_train == 1, n_neg_aug / max(n_pos_aug, 1), 1.0)
    
    for _ in range(lr_epochs):
        z = np.dot(X_train, w_lr) + b_lr
        p = sigmoid(z)
        grad_w = np.dot(X_train.T, (sample_w_lr * (p - y_train))) / N_lr + lr_reg * w_lr
        grad_b = np.mean(sample_w_lr * (p - y_train))
        w_lr -= lr_rate * grad_w
        b_lr -= lr_rate * grad_b
    
    lr_probs = sigmoid(np.dot(X_norm, w_lr) + b_lr)
    lr_thresh, lr_f1 = find_optimal_threshold(lr_probs, y_raw)
    lr_preds = (lr_probs >= lr_thresh).astype(int)
    lr_acc = float(np.mean(lr_preds == y_raw))
    lr_tp = int(np.sum((lr_preds == 1) & (y_raw == 1)))
    lr_tn = int(np.sum((lr_preds == 0) & (y_raw == 0)))
    lr_fp = int(np.sum((lr_preds == 1) & (y_raw == 0)))
    lr_fn = int(np.sum((lr_preds == 0) & (y_raw == 1)))
    lr_sens = float(lr_tp / (lr_tp + lr_fn)) if (lr_tp + lr_fn) > 0 else 0.0
    lr_spec = float(lr_tn / (lr_tn + lr_fp)) if (lr_tn + lr_fp) > 0 else 0.0
    
    print(f"  LR Accuracy: {lr_acc*100:.2f}%, Sens: {lr_sens*100:.2f}%, Spec: {lr_spec*100:.2f}%")
    
    # ============ Ensemble: average MLP + LR probabilities ============
    ensemble_probs = 0.6 * mlp_probs + 0.4 * lr_probs
    ens_thresh, ens_f1 = find_optimal_threshold(ensemble_probs, y_raw)
    ens_preds = (ensemble_probs >= ens_thresh).astype(int)
    
    ens_acc = float(np.mean(ens_preds == y_raw))
    ens_tp = int(np.sum((ens_preds == 1) & (y_raw == 1)))
    ens_tn = int(np.sum((ens_preds == 0) & (y_raw == 0)))
    ens_fp = int(np.sum((ens_preds == 1) & (y_raw == 0)))
    ens_fn = int(np.sum((ens_preds == 0) & (y_raw == 1)))
    ens_sens = float(ens_tp / (ens_tp + ens_fn)) if (ens_tp + ens_fn) > 0 else 0.0
    ens_spec = float(ens_tn / (ens_tn + ens_fp)) if (ens_tn + ens_fp) > 0 else 0.0
    ens_prec = float(ens_tp / (ens_tp + ens_fp)) if (ens_tp + ens_fp) > 0 else 0.0
    ens_f1_final = 2 * ens_prec * ens_sens / (ens_prec + ens_sens) if (ens_prec + ens_sens) > 0 else 0.0
    
    print(f"\n{'='*55}")
    print(f" Ensemble Model Performance (0.6*MLP + 0.4*LR)")
    print(f"{'='*55}")
    print(f"  Accuracy:    {ens_acc * 100:.2f}%")
    print(f"  Sensitivity: {ens_sens * 100:.2f}%")
    print(f"  Specificity: {ens_spec * 100:.2f}%")
    print(f"  F1 Score:    {ens_f1_final * 100:.2f}%")
    print(f"  Threshold:   {ens_thresh}")
    print(f"  Confusion Matrix: TP={ens_tp}, TN={ens_tn}, FP={ens_fp}, FN={ens_fn}")
    print(f"{'='*55}")
    
    # Pick the best model
    if ens_f1_final >= f1:
        print("\n>>> Using ENSEMBLE model (best F1)")
        final_acc, final_sens, final_spec, final_f1 = ens_acc, ens_sens, ens_spec, ens_f1_final
        final_tp, final_tn, final_fp, final_fn = ens_tp, ens_tn, ens_fp, ens_fn
        final_thresh = ens_thresh
        use_ensemble = True
    else:
        print("\n>>> Using MLP model (best F1)")
        final_acc, final_sens, final_spec, final_f1 = accuracy, sens, spec, f1
        final_tp, final_tn, final_fp, final_fn = tp, tn, fp, fn
        final_thresh = optimal_thresh
        use_ensemble = False
    
    # ============ Feature Labels ============
    feature_labels = [
        # Original base features (10)
        'OSDI Symptom Score',
        'Cooling Rate (10S)',
        'Nasal Cornea Temp (°C)',
        'Temporal Cornea Temp (°C)',
        'Central Cornea Temp (°C)',
        'Mean Surface Temp (°C)',
        'Min Surface Temp (°C)',
        'Max Surface Temp (°C)',
        'Thermal Non-Uniformity (Std)',
        'Tear Breakup Cold Spots Ratio',
        # New raw features (8)
        'Superior Cornea Temp (°C)',
        'Inferior Cornea Temp (°C)',
        'Median Surface Temp (°C)',
        'P10 Surface Temp (°C)',
        'Interquartile Range (°C)',
        'Temperature Skewness',
        'Temperature Kurtosis',
        'Warm Spots Ratio',
        # Derived clinical features (14+)
        'Nasal-Temporal Gradient (°C)',
        'Superior-Inferior Gradient (°C)',
        'Central Cornea-Mean Diff (°C)',
        'Thermal Range (°C)',
        'Central Cornea-Median Diff (°C)',
        'OSDI High Flag',
        'OSDI Moderate Flag',
        'OSDI × Cornea Cooling',
        'OSDI × Cold Spots',
        'Cooling Rate Magnitude',
        'Negative Cooling Flag',
        'Cold-Warm Balance',
        'Low Central Cornea Flag',
        'Lower Tail Spread',
        'Upper Tail Spread',
    ]
    
    # ============ Export ============
    # Export MLP params as serializable lists
    W1, b1, W2, b2, W3, b3 = mlp_params
    
    export_payload = {
        'model_type': 'Thermal Anterior Segment AI Classifier (MLP Ensemble)',
        'model_architecture': 'MLP(32,16) + Logistic Regression Ensemble',
        # MLP weights
        'mlp': {
            'W1': W1.tolist(),
            'b1': b1.tolist(),
            'W2': W2.tolist(),
            'b2': b2.tolist(),
            'W3': W3.flatten().tolist(),
            'b3': float(b3[0]),
        },
        # LR weights (for ensemble and backwards compatibility)
        'weights': w_lr.tolist(),
        'bias': float(b_lr),
        'ensemble_weight_mlp': 0.6,
        'ensemble_weight_lr': 0.4,
        'use_ensemble': use_ensemble,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'threshold': final_thresh,
        'feature_labels': feature_labels,
        'metrics': {
            'accuracy': final_acc,
            'sensitivity': final_sens,
            'specificity': final_spec,
            'precision': ens_prec if use_ensemble else precision,
            'f1_score': final_f1,
            'tp': final_tp,
            'tn': final_tn,
            'fp': final_fp,
            'fn': final_fn,
            'total_samples': N_total,
            'mlp_accuracy': accuracy,
            'lr_accuracy': lr_acc,
            'ensemble_accuracy': ens_acc,
        },
        'sample_gallery': sample_gallery
    }
    
    with open('thermal_model_data.json', 'w', encoding='utf-8') as f:
        json.dump(export_payload, f, indent=2)
        
    print(f"\nSuccessfully exported thermal model to thermal_model_data.json!")
    print(f"Saved {len(sample_gallery)} web-accessible thermal image samples into /samples/")
    print(f"\nFinal Model: Accuracy={final_acc*100:.1f}%, Sensitivity={final_sens*100:.1f}%, Specificity={final_spec*100:.1f}%, F1={final_f1*100:.1f}%")

if __name__ == '__main__':
    train_and_export_thermal_system()
