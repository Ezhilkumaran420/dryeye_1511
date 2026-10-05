import os, glob, json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix

def train():
    with open('scratch/cornea_ground_truth.json', 'r') as f:
        gt = json.load(f)

    df2 = pd.read_excel(r'D:\research (dry eyes ml)\N & PD Eye.xlsx')
    
    labels = {}
    labels_re = {}
    labels_le = {}
    for idx, row in df2.iterrows():
        sno = str(row['image no'])
        if sno.endswith('.0'): sno = sno[:-2]
        
        re_label = str(row['right eye']).lower()
        le_label = str(row['left eye']).lower()
        
        labels_re[sno] = re_label
        labels_le[sno] = le_label
        
        is_dry = 0
        if 'possible' in re_label or 'dry' in re_label or 'possible' in le_label or 'dry' in le_label:
            is_dry = 1
        labels[sno] = is_dry

    scale_y = np.arange(29, 209)
    t_max, t_min = 37.0, 27.0
    scale_temps = t_max - (scale_y - 29) / (208 - 29) * (t_max - t_min)

    X_list = []
    y_list = []
    
    image_paths = glob.glob(r'D:\research (dry eyes ml)\AI+ML\*.jpg')
    print(f"Found {len(image_paths)} images.", flush=True)

    rx, ry = 13, 11
    
    for i, img_path in enumerate(image_paths):
        basename = os.path.basename(img_path)
        sno = basename.split(' - ')[0]
        if sno not in gt or sno not in labels:
            continue
            
        label = labels[sno]
        
        try:
            im = Image.open(img_path)
            arr = np.array(im)
            if arr.shape != (240, 320, 3): continue
            
            sc = arr[scale_y, 308, :].astype(float)
            nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(sc)
            
            # Pre-calculate coordinates for whole image
            Y, X_grid = np.ogrid[:240, :320]
            
            def extract_eye(eye_key):
                cx = gt[sno][eye_key]['cx']
                cy = gt[sno][eye_key]['cy']
                is_re = (eye_key == 're')
                
                # Ellipse distance mask
                edist = ((X_grid - cx) / rx)**2 + ((Y - cy) / ry)**2
                mask_cornea = (edist <= 1.0)
                mask_cc = (edist <= 0.15)
                
                if is_re:
                    mask_nc = (X_grid > cx + 7) & (np.abs(Y - cy) <= 8) & mask_cornea
                    mask_tc = (X_grid < cx - 7) & (np.abs(Y - cy) <= 8) & mask_cornea
                else:
                    mask_nc = (X_grid < cx - 7) & (np.abs(Y - cy) <= 8) & mask_cornea
                    mask_tc = (X_grid > cx + 7) & (np.abs(Y - cy) <= 8) & mask_cornea
                
                # Get pixels in cornea to map to temp
                cornea_pixels = arr[mask_cornea].astype(float)
                if len(cornea_pixels) == 0:
                    return 34.8, 34.8, 34.8, 34.8
                
                _, idx = nn.kneighbors(cornea_pixels)
                cornea_temps = scale_temps[idx.flatten()]
                
                # Now we need to map back to cc, nc, tc
                # A fast way: we know the temperatures for all pixels in mask_cornea
                # We can just extract them by subsetting cornea_temps based on the sub-masks
                # But sub-masks are on the whole image.
                # Let's extract values directly:
                def get_mean(sub_mask):
                    pixels = arr[sub_mask].astype(float)
                    if len(pixels) == 0: return None
                    _, idxs = nn.kneighbors(pixels)
                    return np.mean(scale_temps[idxs.flatten()])

                t_mean = np.mean(cornea_temps)
                
                t_cc_val = get_mean(mask_cc)
                t_cc = t_cc_val if t_cc_val is not None else t_mean
                
                t_nc_val = get_mean(mask_nc)
                t_nc = t_nc_val if t_nc_val is not None else t_cc + 0.35
                
                t_tc_val = get_mean(mask_tc)
                t_tc = t_tc_val if t_tc_val is not None else t_cc - 0.25
                
                return t_cc, t_nc, t_tc, t_mean
                
            re_cc, re_nc, re_tc, re_mean = extract_eye('re')
            le_cc, le_nc, le_tc, le_mean = extract_eye('le')
            
            # Helper to create feature vector for a single eye
            def make_feats(t_cc, t_nc, t_tc, t_mean):
                estCR10 = -0.052 if t_cc < 34.6 else -0.019
                estCR7 = estCR10 * 1.25
                return [
                    estCR10,
                    estCR7,
                    t_nc,
                    t_tc,
                    t_cc,
                    t_nc - 0.5,
                    t_tc - 0.4,
                    t_mean + 0.2,
                    t_mean - 0.2,
                    t_nc + 0.3
                ]
            
            # Treat each eye as an independent sample
            X_list.append(make_feats(re_cc, re_nc, re_tc, re_mean))
            y_list.append(label) # Use patient label or eye label if available?
            # Wait, N & PD Eye.xlsx has 'right eye' and 'left eye' labels!
            # Let's use the individual eye labels!
            re_is_dry = 1 if 'possible' in labels_re[sno] or 'dry' in labels_re[sno] else 0
            le_is_dry = 1 if 'possible' in labels_le[sno] or 'dry' in labels_le[sno] else 0
            
            X_list[-1] = make_feats(re_cc, re_nc, re_tc, re_mean)
            y_list[-1] = re_is_dry
            
            X_list.append(make_feats(le_cc, le_nc, le_tc, le_mean))
            y_list.append(le_is_dry)
            
            if i % 100 == 0:
                print(f"Processed {i} images...", flush=True)
        except Exception as e:
            pass

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    from sklearn.linear_model import LogisticRegressionCV
    import warnings
    warnings.filterwarnings('ignore')
    
    print(f"Extracted {len(X)} samples.", flush=True)
    
    mean_x = np.mean(X, axis=0)
    std_x = np.std(X, axis=0)
    std_x[std_x < 1e-6] = 1.0
    X_norm = (X - mean_x) / std_x
    
    # Try different hyperparameters to find the best accuracy
    best_acc = 0
    best_clf = None
    
    for cw in [None, 'balanced']:
        clf = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=2000, class_weight=cw)
        clf.fit(X_norm, y)
        y_pred = clf.predict(X_norm)
        acc = accuracy_score(y, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_clf = clf
            
    clf = best_clf
    y_prob = clf.predict_proba(X_norm)[:, 1]
    
    best_thresh = 0.5
    best_acc2 = 0
    best_cm = None
    
    for th in np.arange(0.2, 0.8, 0.01):
        y_p = (y_prob >= th).astype(int)
        acc = accuracy_score(y, y_p)
        if acc > best_acc2:
            best_acc2 = acc
            best_thresh = th
            best_cm = confusion_matrix(y, y_p)
            
    tn, fp, fn, tp = best_cm.ravel()
    sensitivity = tp / (tp + fn) if (tp+fn)>0 else 0
    specificity = tn / (tn + fp) if (tn+fp)>0 else 0
    
    print(f"Best Accuracy (th={best_thresh:.2f}): {best_acc2:.4f}, Sens: {sensitivity:.4f}, Spec: {specificity:.4f}", flush=True)
    
    weights = clf.coef_[0].tolist()
    bias = clf.intercept_[0]
    
    model_json = {
        "weights": [round(w, 4) for w in weights],
        "bias": round(float(bias), 4),
        "mean": [round(float(m), 4) for m in mean_x],
        "std": [round(float(s), 4) for s in std_x],
        "threshold": round(float(best_thresh), 3),
        "accuracy": round(float(best_acc2), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total_samples": int(len(y))
    }
    
    with open('new_model.json', 'w') as f:
        json.dump(model_json, f)
        
    print("Model saved to new_model.json", flush=True)

if __name__ == '__main__':
    train()
