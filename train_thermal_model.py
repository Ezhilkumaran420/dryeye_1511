import os, re, glob, json, shutil
import openpyxl
from PIL import Image
import numpy as np

def extract_thermal_metrics(img_path):
    """
    Extracts calibrated ocular surface temperature metrics from an anterior segment FLIR thermal image.
    Uses scale bar color calibration and anatomical ocular ROI segmentation.
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
            
            # Central cornea (center patch)
            cy, cx = h // 2, w // 2
            r_y, r_x = 12, 16
            cc_patch = temps[max(0, cy-r_y):min(h, cy+r_y), max(0, cx-r_x):min(w, cx+r_x)]
            
            # Nasal cornea (inner side: for RE this is right, for LE this is left)
            nc_patch = temps[max(0, cy-10):min(h, cy+10), min(w-20, cx+15):min(w, cx+35)]
            # Temporal cornea (outer side: for RE this is left, for LE this is right)
            tc_patch = temps[max(0, cy-10):min(h, cy+10), max(0, cx-35):max(15, cx-15)]
            
            t_cc = float(np.mean(cc_patch)) if cc_patch.size > 0 else float(np.mean(temps))
            t_nc = float(np.mean(nc_patch)) if nc_patch.size > 0 else t_cc + 0.5
            t_tc = float(np.mean(tc_patch)) if tc_patch.size > 0 else t_cc - 0.5
            t_mean = float(np.mean(temps))
            t_min_val = float(np.percentile(temps, 5))
            t_max_val = float(np.percentile(temps, 95))
            t_std = float(np.std(temps))
            
            # Cold spots indicating localized tear film breakup (pixels < 34.2°C)
            cold_spots_ratio = float(np.mean(temps < 34.2))
            
            return {
                't_cc': round(t_cc, 2),
                't_nc': round(t_nc, 2),
                't_tc': round(t_tc, 2),
                't_mean': round(t_mean, 2),
                't_min': round(t_min_val, 2),
                't_max': round(t_max_val, 2),
                't_std': round(t_std, 2),
                'cold_spots_ratio': round(cold_spots_ratio, 3)
            }
            
        re_metrics = process_eye_roi(arr[110:170, 50:140])
        le_metrics = process_eye_roi(arr[110:170, 175:265])
        
        return {
            're': re_metrics,
            'le': le_metrics
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def train_and_export_thermal_system():
    thermal_dir = r"D:\AI+ML"
    excel_path = r"C:\Users\ezhil\Downloads\Eye classification.xlsx"
    npd_path = r"C:\Users\ezhil\Downloads\N & PD Eye.xlsx"
    
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
        
        # Right Eye entry
        re_data = re_clinical[sno]
        subjects_dataset.append({
            'sno': sno,
            'eye': 'RE',
            'label': re_data['label'],
            'label_str': re_data['label_str'],
            'features': [
                re_data['osdi'],
                re_cr,
                metrics0['re']['t_nc'],
                metrics0['re']['t_tc'],
                metrics0['re']['t_cc'],
                metrics0['re']['t_mean'],
                metrics0['re']['t_min'],
                metrics0['re']['t_max'],
                metrics0['re']['t_std'],
                metrics0['re']['cold_spots_ratio']
            ]
        })
        
        # Left Eye entry
        le_data = le_clinical.get(sno, re_data)
        subjects_dataset.append({
            'sno': sno,
            'eye': 'LE',
            'label': le_data['label'],
            'label_str': le_data['label_str'],
            'features': [
                le_data['osdi'],
                le_cr,
                metrics0['le']['t_nc'],
                metrics0['le']['t_tc'],
                metrics0['le']['t_cc'],
                metrics0['le']['t_mean'],
                metrics0['le']['t_min'],
                metrics0['le']['t_max'],
                metrics0['le']['t_std'],
                metrics0['le']['cold_spots_ratio']
            ]
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
    
    # Train Logistic Regression Classifier on Image Features
    X = np.array([s['features'] for s in subjects_dataset])
    y = np.array([s['label'] for s in subjects_dataset])
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std
    
    N, D = X_norm.shape
    w = np.zeros(D)
    b = 0.0
    lr = 0.15
    epochs = 1200
    reg = 0.04
    
    for _ in range(epochs):
        z = np.dot(X_norm, w) + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
        grad_w = np.dot(X_norm.T, (p - y)) / N + reg * w
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
        
    probs = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X_norm, w) + b, -15.0, 15.0)))
    preds = (probs >= 0.40).astype(int)
    
    accuracy = float(np.mean(preds == y))
    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    print(f"\nThermal Image Model Performance (N={N}):")
    print(f"  Accuracy:    {accuracy * 100:.2f}%")
    print(f"  Sensitivity: {sens * 100:.2f}%")
    print(f"  Specificity: {spec * 100:.2f}%")
    print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    feature_labels = [
        'OSDI Symptom Score',
        'Cooling Rate (10S)',
        'Nasal Cornea Temp (°C)',
        'Temporal Cornea Temp (°C)',
        'Central Cornea Temp (°C)',
        'Mean Surface Temp (°C)',
        'Min Surface Temp (°C)',
        'Max Surface Temp (°C)',
        'Thermal Non-Uniformity (Std)',
        'Tear Breakup Cold Spots Ratio'
    ]
    
    export_payload = {
        'model_type': 'Thermal Anterior Segment AI Classifier',
        'weights': w.tolist(),
        'bias': float(b),
        'mean': mean.tolist(),
        'std': std.tolist(),
        'threshold': 0.40,
        'feature_labels': feature_labels,
        'metrics': {
            'accuracy': accuracy,
            'sensitivity': sens,
            'specificity': spec,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total_samples': N
        },
        'sample_gallery': sample_gallery
    }
    
    with open('thermal_model_data.json', 'w', encoding='utf-8') as f:
        json.dump(export_payload, f, indent=2)
        
    print("Successfully exported thermal model to thermal_model_data.json!")
    print(f"Saved {len(sample_gallery)} web-accessible thermal image samples into /samples/")

if __name__ == '__main__':
    train_and_export_thermal_system()
