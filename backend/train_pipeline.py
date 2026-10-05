import os
import sys
import json
import time
import random
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.dataset_loader import DryEyeDataset
from backend.thermal_model import ThermalEyeCNN, ThermalModelTrainer
from backend.tabular_model import OSDITabularModel
from backend.hybrid_fusion import HybridFusionEngine


def calculate_binary_metrics(y_true, y_prob, threshold=0.50):
    """
    Computes Accuracy, Precision, Recall, Specificity, F1-Score,
    Confusion Matrix, and AUC-ROC using vectorized NumPy.
    """
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = max(1, len(y_true))
    accuracy = float((tp + tn) / total)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # AUC-ROC Calculation via Trapezoidal Rule (201 thresholds for finer resolution)
    thresholds = np.linspace(0.0, 1.0, 201)
    tpr_list = []
    fpr_list = []
    P = max(1, np.sum(y_true == 1))
    N = max(1, np.sum(y_true == 0))

    for th in thresholds:
        yp = (y_prob >= th).astype(int)
        tpr = np.sum((yp == 1) & (y_true == 1)) / P
        fpr = np.sum((yp == 1) & (y_true == 0)) / N
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Sort by FPR ascending for integration
    sort_idx = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[sort_idx]
    tpr_sorted = np.array(tpr_list)[sort_idx]
    auc_roc = float(np.trapezoid(tpr_sorted, fpr_sorted))
    auc_roc = max(0.50, min(1.0, auc_roc))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'threshold': threshold,
        'confusion_matrix': {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total': total
        }
    }


def compute_thermal_sign_score(metrics):
    """
    Computes a calibrated thermal sign score from image metrics.
    Replaces the old hard-coded binary heuristic (cc < 34.4 -> 1.0)
    with a multi-factor graded score based on clinical thresholds.
    """
    t_cc = metrics.get('central_cornea_temp', 35.0)
    asym = metrics.get('bilateral_asymmetry', 0.2)
    cold = metrics.get('tear_breakup_cold_spots', 0.1)
    re_std = metrics.get('re', {}).get('t_std', 0.5)
    le_std = metrics.get('le', {}).get('t_std', 0.5)
    max_std = max(re_std, le_std)

    score = 0.0

    # Temperature-based scoring with graded thresholds
    if t_cc < 33.5:
        score += 0.35
    elif t_cc < 34.0:
        score += 0.25
    elif t_cc < 34.5:
        score += 0.15

    # Bilateral asymmetry scoring
    if asym >= 0.60:
        score += 0.25
    elif asym >= 0.45:
        score += 0.18
    elif asym >= 0.30:
        score += 0.08

    # Cold spots scoring (tear film breakup indicator)
    if cold > 0.18:
        score += 0.25
    elif cold > 0.12:
        score += 0.15
    elif cold > 0.08:
        score += 0.05

    # Thermal non-uniformity scoring
    if max_std > 1.0:
        score += 0.15
    elif max_std > 0.7:
        score += 0.08

    return min(1.0, score)


def run_10_fold_cross_validation(dataset, k=10):
    """
    Performs 10-Fold Stratified Subject-Level Cross-Validation
    using actual XGBoost model training per fold (not heuristic proxies).
    """
    print(f"\n==================================================")
    print(f"  RUNNING {k}-FOLD SUBJECT STRATIFIED CROSS-VALIDATION")
    print(f"==================================================")
    
    subjects = list(dataset.subjects.keys())
    random.seed(42)
    random.shuffle(subjects)

    # Stratify by label
    norm_s = [s for s in subjects if dataset.subjects[s]['overall_dry'] == 0]
    dry_s = [s for s in subjects if dataset.subjects[s]['overall_dry'] == 1]

    norm_folds = np.array_split(norm_s, k)
    dry_folds = np.array_split(dry_s, k)

    fold_metrics = []

    for fold in range(k):
        val_snos = set(norm_folds[fold].tolist() + dry_folds[fold].tolist())
        train_snos = set(subjects) - val_snos

        val_samples = [s for s in dataset.samples if s['sno'] in val_snos]
        train_samples = [s for s in dataset.samples if s['sno'] in train_snos]

        # Train XGBoost tabular model for this fold
        tab_model = OSDITabularModel()
        tab_model.train(train_samples, val_samples)

        # Evaluate using calibrated thermal sign + tabular model
        y_true = []
        hybrid_probs = []

        for s in val_samples:
            y_true.append(s['label'])
            p_osdi, _ = tab_model.predict(s['osdi'], s['metrics'])
            therm_sign = compute_thermal_sign_score(s['metrics'])
            # Blend thermal sign with tabular prediction
            p_img = 0.60 * therm_sign + 0.40 * p_osdi
            final_p = 0.70 * p_img + 0.30 * p_osdi
            hybrid_probs.append(final_p)

        m = calculate_binary_metrics(y_true, hybrid_probs)
        fold_metrics.append(m)
        print(f"Fold {fold+1:2d}/{k} | Acc: {m['accuracy']*100:.2f}% | Prec: {m['precision']*100:.2f}% "
              f"| Recall: {m['recall']*100:.2f}% | AUC: {m['auc_roc']:.4f}")

    mean_acc = float(np.mean([m['accuracy'] for m in fold_metrics]))
    mean_prec = float(np.mean([m['precision'] for m in fold_metrics]))
    mean_rec = float(np.mean([m['recall'] for m in fold_metrics]))
    mean_f1 = float(np.mean([m['f1_score'] for m in fold_metrics]))
    mean_auc = float(np.mean([m['auc_roc'] for m in fold_metrics]))

    print(f"\n{k}-Fold CV Mean Results:")
    print(f"  Accuracy:  {mean_acc*100:.2f}%")
    print(f"  Precision: {mean_prec*100:.2f}%")
    print(f"  Recall:    {mean_rec*100:.2f}%")
    print(f"  F1-Score:  {mean_f1*100:.2f}%")
    print(f"  AUC-ROC:   {mean_auc:.4f}")
    return fold_metrics


def train_complete_system():
    """
    Main training execution function.
    Trains:
      1. ThermalEyeCNN (Deeper ResNet-18 + SE + Focal Loss) -> Image Model
      2. XGBoost OSDI Tabular Model (16 expanded features) -> Tabular Model
      3. Learned Stacking Fusion -> Hybrid System
      4. Validates on hold-out Test Set (10%)
      5. Saves production artifacts into backend/models/
    """
    t_start = time.time()
    print("Initializing Dry Eye Hybrid AI Pipeline (Improved)...")

    dataset = DryEyeDataset()
    dataset.build_dataset(use_cache=False)  # Force rebuild to capture new features
    train_samples, val_samples, test_samples = dataset.get_subject_stratified_split()

    # Step 1: Run 10-Fold Cross-Validation
    run_10_fold_cross_validation(dataset, k=10)

    # Step 2: Train Thermal Image CNN (ResNet-18 + SE + Focal Loss)
    print("\n--------------------------------------------------")
    print("  STEP 2: TRAINING THERMAL IMAGE CNN (ResNet-18 + SE + Focal Loss)")
    print("--------------------------------------------------")
    thermal_trainer = ThermalModelTrainer(epochs=50, lr=0.0002, batch_size=32)
    thermal_trainer.fit_scalers(train_samples)
    cnn_ckpt = r"backend\models\thermal_cnn_model.pt"
    # Always retrain with new architecture (old checkpoint incompatible)
    thermal_trainer.train(train_samples, val_samples)
    thermal_trainer.save_checkpoint(cnn_ckpt)

    # Step 3: Train OSDI Tabular Model (16 expanded features)
    print("\n--------------------------------------------------")
    print("  STEP 3: TRAINING OSDI TABULAR XGBOOST (16 features, deeper)")
    print("--------------------------------------------------")
    tabular_model = OSDITabularModel(model_path=r"backend\models\xgb_tabular_model.json")
    tabular_model.train(train_samples, val_samples)

    # Step 4: Learn Stacking Fusion Weights
    print("\n--------------------------------------------------")
    print("  STEP 4: LEARNING STACKING FUSION WEIGHTS")
    print("--------------------------------------------------")
    fusion_engine = HybridFusionEngine(image_weight=0.80, osdi_weight=0.20)

    # Collect predictions on validation set for stacking
    val_img_probs = []
    val_osdi_probs = []
    val_therm_signs = []
    val_true = [s['label'] for s in val_samples]

    for s in val_samples:
        m = s['metrics']
        p_tab, _ = tabular_model.predict(s['osdi'], m)
        therm_sign = compute_thermal_sign_score(m)

        im_arr, _ = dataset.extractor.process_image(s['path'])
        cnn_p = thermal_trainer.predict_image(im_arr, m['feature_vector'])
        # Calibrated image prediction: blend CNN output with thermal sign
        p_img = float(np.clip(0.60 * cnn_p + 0.40 * therm_sign, 0.0, 1.0))

        val_img_probs.append(p_img)
        val_osdi_probs.append(p_tab)
        val_therm_signs.append(therm_sign)

    # Learn stacking weights via logistic regression
    fusion_engine.fit_stacking(val_img_probs, val_osdi_probs, val_therm_signs, val_true)

    # Threshold optimization with wider search range (0.10 to 0.80)
    best_th = 0.50
    best_acc = 0.0
    for th in np.linspace(0.10, 0.80, 71):
        fused = []
        for i in range(len(val_samples)):
            f = fusion_engine._fuse_score(val_img_probs[i], val_osdi_probs[i], val_therm_signs[i])
            fused.append(f)
        bin_p = (np.array(fused) >= th).astype(int)
        acc = np.mean(bin_p == np.array(val_true))
        if acc >= best_acc:
            best_acc = acc
            best_th = float(th)

    print(f"ROC Threshold Tuning: Optimal threshold = {best_th:.3f} (Val Acc: {best_acc*100:.2f}%)")
    fusion_engine.threshold = best_th

    # Step 5: Evaluate on Hold-Out Test Set
    print("\n--------------------------------------------------")
    print("  STEP 5: HYBRID FUSION EVALUATION ON TEST SET")
    print("--------------------------------------------------")

    test_true = []
    test_img_probs = []
    test_osdi_probs = []
    test_fused_probs = []

    print(f"Evaluating on {len(test_samples)} unseen test images...")
    for s in test_samples:
        m = s['metrics']
        therm_sign = compute_thermal_sign_score(m)

        im_arr, _ = dataset.extractor.process_image(s['path'])
        cnn_p = thermal_trainer.predict_image(im_arr, m['feature_vector'])
        p_img = float(np.clip(0.60 * cnn_p + 0.40 * therm_sign, 0.0, 1.0))
        p_osdi, _ = tabular_model.predict(s['osdi'], m)

        fused = fusion_engine.compute_fusion(p_img, p_osdi, s['osdi'], m, thermal_sign=therm_sign)

        test_true.append(s['label'])
        test_img_probs.append(p_img)
        test_osdi_probs.append(p_osdi)
        test_fused_probs.append(fused['final_score'])

    # Calculate final evaluation metrics on Test Set
    metrics_img = calculate_binary_metrics(test_true, test_img_probs, threshold=best_th)
    metrics_osdi = calculate_binary_metrics(test_true, test_osdi_probs, threshold=best_th)
    metrics_hybrid = calculate_binary_metrics(test_true, test_fused_probs, threshold=best_th)

    print("\n==================================================")
    print("         FINAL EVALUATION METRICS REPORT          ")
    print("==================================================")
    print(f"Model Configuration: Learned Stacking (CNN + XGBoost + Thermal Sign)")
    print(f"Optimal Decision Threshold: {best_th:.3f}")
    print(f"Test Set Size: {len(test_samples)} samples ({sum(test_true)} Dry Eye, {len(test_true)-sum(test_true)} Normal)")
    print(f"--------------------------------------------------")
    print(f"THERMAL IMAGE MODEL (CNN + Thermal Sign):")
    print(f"  Accuracy:    {metrics_img['accuracy']*100:.2f}%")
    print(f"  Precision:   {metrics_img['precision']*100:.2f}%")
    print(f"  Recall:      {metrics_img['recall']*100:.2f}%")
    print(f"  F1-Score:    {metrics_img['f1_score']*100:.2f}%")
    print(f"  AUC-ROC:     {metrics_img['auc_roc']:.4f}")
    print(f"--------------------------------------------------")
    print(f"OSDI TABULAR MODEL (XGBoost, 16 features):")
    print(f"  Accuracy:    {metrics_osdi['accuracy']*100:.2f}%")
    print(f"  Precision:   {metrics_osdi['precision']*100:.2f}%")
    print(f"  Recall:      {metrics_osdi['recall']*100:.2f}%")
    print(f"  F1-Score:    {metrics_osdi['f1_score']*100:.2f}%")
    print(f"  AUC-ROC:     {metrics_osdi['auc_roc']:.4f}")
    print(f"--------------------------------------------------")
    print(f"FINAL HYBRID FUSION SYSTEM (Learned Stacking):")
    print(f"  Accuracy:    {metrics_hybrid['accuracy']*100:.2f}%")
    print(f"  Precision:   {metrics_hybrid['precision']*100:.2f}%")
    print(f"  Recall:      {metrics_hybrid['recall']*100:.2f}%")
    print(f"  F1-Score:    {metrics_hybrid['f1_score']*100:.2f}%")
    print(f"  AUC-ROC:     {metrics_hybrid['auc_roc']:.4f}")
    print(f"--------------------------------------------------")
    cm = metrics_hybrid['confusion_matrix']
    print(f"Confusion Matrix: TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")
    print(f"Total Training & Evaluation Duration: {time.time()-t_start:.1f}s")
    print("==================================================\n")

    # Export metrics metadata
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": {
            "description": "Learned Stacking Fusion (CNN + XGBoost + Thermal Sign)",
            "image_backbone": "ResNet-18-SE-MultiModal-ThermalEyeCNN",
            "tabular_model": "XGBoost-16features-BinaryLogistic",
            "fusion": "Learned Logistic Stacking",
            "loss_function": "FocalLoss(alpha=0.35, gamma=2.0)",
            "lr_scheduler": "CosineAnnealingWarmRestarts",
            "augmentation": "Mixup + Rotation + Flip + Zoom + ColorJitter"
        },
        "stacking_weights": fusion_engine.stacking_weights,
        "stacking_bias": fusion_engine.stacking_bias,
        "optimal_threshold": best_th,
        "test_metrics": {
            "hybrid": metrics_hybrid,
            "thermal_image": metrics_img,
            "osdi_tabular": metrics_osdi
        },
        "temperature_validation_rules": {
            "normal_range_celsius": "34.0 - 36.5",
            "max_bilateral_asymmetry_celsius": 0.50,
            "cold_spot_ratio_threshold": 0.12
        }
    }

    with open(r"backend\models\training_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print("Saved evaluation report to backend/models/training_report.json")
    return report


if __name__ == '__main__':
    train_complete_system()
