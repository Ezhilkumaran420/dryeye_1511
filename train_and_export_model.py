import openpyxl
import numpy as np
import json
import os

def load_and_process_dataset(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    all_patients = []
    
    # Feature columns to extract
    feature_keys = [
        'OSDI',
        'Cooling rate of roi 10S',
        'Cooling rate of roi 7S',
        '3. NC',
        '4. TC',
        '7. CC',
        '5. NL',
        '6. TL',
        '0s',
        '10s',
        'MOST (from all the intervals)',
        'N',
        'C',
        'T'
    ]
    
    feature_labels = [
        'OSDI Score',
        'Cooling Rate (10s)',
        'Cooling Rate (7s)',
        'Nasal Cornea (°C)',
        'Temporal Cornea (°C)',
        'Central Cornea (°C)',
        'Nasal Limbus (°C)',
        'Temporal Limbus (°C)',
        'Temp at 0s (°C)',
        'Temp at 10s (°C)',
        'MOST (°C)',
        'Nasal Temp (°C)',
        'Central Temp (°C)',
        'Temporal Temp (°C)'
    ]

    for eye_type in ['RE', 'LE']:
        ws = wb[eye_type]
        rows = list(ws.iter_rows(values_only=True))
        headers = rows[0]
        
        # Map indices
        indices = {}
        for k in feature_keys:
            if k in headers:
                indices[k] = headers.index(k)
        
        target_idx = -1
        sno_idx = headers.index('S.No') if 'S.No' in headers else 0
        age_idx = headers.index('Age') if 'Age' in headers else None
        gender_idx = headers.index('Gender') if 'Gender' in headers else None
        
        for r in rows[1:]:
            target_val = str(r[target_idx] or '').strip().lower()
            if 'normal' in target_val:
                label = 0
                label_str = 'Normal'
            elif 'dry' in target_val:
                label = 1
                label_str = 'Possible Dry Eye'
            else:
                continue
                
            patient_sno = r[sno_idx]
            age = r[age_idx] if age_idx is not None and age_idx < len(r) else None
            gender = r[gender_idx] if gender_idx is not None and gender_idx < len(r) else None
            
            features = []
            for k in feature_keys:
                idx = indices.get(k)
                val = r[idx] if idx is not None and idx < len(r) else 0.0
                try:
                    features.append(float(val) if val is not None else 0.0)
                except (ValueError, TypeError):
                    features.append(0.0)
            
            all_patients.append({
                'sno': patient_sno,
                'eye': eye_type,
                'age': age,
                'gender': gender,
                'label': label,
                'label_str': label_str,
                'features': features
            })
            
    return all_patients, feature_keys, feature_labels

def train_classifier(patients):
    X = np.array([p['features'] for p in patients])
    y = np.array([p['label'] for p in patients])
    
    # Feature standardization
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    X_norm = (X - mean) / std
    
    # Logistic Regression with L2 regularization
    N, D = X_norm.shape
    w = np.zeros(D)
    b = 0.0
    lr = 0.2
    epochs = 1200
    reg = 0.03
    
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
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    print(f"Overall Dataset Performance (N={N}):")
    print(f"  Accuracy:    {accuracy * 100:.2f}%")
    print(f"  Sensitivity: {sensitivity * 100:.2f}%")
    print(f"  Specificity: {specificity * 100:.2f}%")
    print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    return {
        'weights': w.tolist(),
        'bias': float(b),
        'mean': mean.tolist(),
        'std': std.tolist(),
        'threshold': 0.40,
        'metrics': {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total_samples': N,
            'normal_count': int(np.sum(y == 0)),
            'dry_count': int(np.sum(y == 1))
        }
    }

def main():
    excel_path = r"C:\Users\ezhil\Downloads\Eye classification.xlsx"
    if not os.path.exists(excel_path):
        print("Dataset not found at:", excel_path)
        return
        
    patients, feature_keys, feature_labels = load_and_process_dataset(excel_path)
    print(f"Loaded {len(patients)} eye samples.")
    
    model_data = train_classifier(patients)
    
    # Calculate group averages for clinical reference
    X_norm_group = np.array([p['features'] for p in patients if p['label'] == 0])
    X_dry_group = np.array([p['features'] for p in patients if p['label'] == 1])
    
    reference_stats = {
        'normal_mean': np.mean(X_norm_group, axis=0).tolist(),
        'dry_mean': np.mean(X_dry_group, axis=0).tolist(),
        'normal_std': np.std(X_norm_group, axis=0).tolist(),
        'dry_std': np.std(X_dry_group, axis=0).tolist()
    }
    
    # Build a compact sample cohort for UI live testing (20 representative subjects)
    sample_cohort = []
    # Pick a balanced selection across right and left eyes
    for p in patients:
        sample_cohort.append({
            'id': f"Subject #{p['sno']} ({p['eye']})",
            'sno': p['sno'],
            'eye': p['eye'],
            'actual_diagnosis': p['label_str'],
            'features': {feature_keys[i]: round(p['features'][i], 3) for i in range(len(feature_keys))}
        })
        
    export_payload = {
        'feature_keys': feature_keys,
        'feature_labels': feature_labels,
        'model': model_data,
        'reference_stats': reference_stats,
        'sample_cohort': sample_cohort[:30] # 30 patients for quick UI testing
    }
    
    out_file = "model_data.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
        
    print(f"Successfully exported model and dataset statistics to {out_file}")

if __name__ == "__main__":
    main()
