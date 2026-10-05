import os
import json
import numpy as np
import xgboost as xgb

class OSDITabularModel:
    """
    XGBoost + Clinical OSDI Severity Model with expanded features (16-dim):
      - Maps OSDI score (0-100) to severity grades:
          * 0-12   -> Normal (0)
          * 13-22  -> Mild Dry Eye (1)
          * 23-32  -> Moderate Dry Eye (2)
          * 33-100 -> Severe Dry Eye (3)
      - Fuses OSDI score with 14 image-derived thermal features
        (all available at inference time from a single thermal image).
      - Uses deeper XGBoost with class imbalance handling via scale_pos_weight.
      - Outputs calibrated probability of Dry Eye (0.0 to 1.0).
    """
    def __init__(self, model_path=r"backend\models\xgb_tabular_model.json"):
        self.model_path = model_path
        self.booster = None
        self.feature_names = [
            'osdi',
            'osdi_severity_idx',
            'central_cornea_temp',
            'cooling_rate_10s',
            'bilateral_asymmetry',
            'tear_breakup_cold_spots',
            're_tmean',
            're_tstd',
            're_tcc',
            're_tnc',
            're_ttc',
            'le_tcc',
            'le_tmean',
            'le_tstd',
            're_nt_gradient',
            'le_cold_spots'
        ]
        # Clinical population reference values for imputation when data is absent
        self.default_clinical_means = {
            'central_cornea_temp': 34.65,
            'cooling_rate_10s': -0.035,
            'bilateral_asymmetry': 0.35,
            'tear_breakup_cold_spots': 0.12,
            're_tmean': 35.10,
            're_tstd': 0.72,
            're_tcc': 34.65,
            're_tnc': 35.00,
            're_ttc': 34.30,
            'le_tcc': 34.65,
            'le_tmean': 35.10,
            'le_tstd': 0.72,
            're_nt_gradient': 0.70,
            'le_cold_spots': 0.12
        }

    @staticmethod
    def get_osdi_severity(score):
        """
        Maps OSDI score (0-100) to clinical severity tuple: (index, label)
        """
        s = float(score) if score is not None else 0.0
        s = max(0.0, min(100.0, s))
        if s <= 12.0:
            return 0, "Normal"
        elif s <= 22.0:
            return 1, "Mild Dry Eye"
        elif s <= 32.0:
            return 2, "Moderate Dry Eye"
        else:
            return 3, "Severe Dry Eye"

    def train(self, train_samples, val_samples):
        """
        Trains native XGBoost on the expanded 16-feature tabular clinical dataset.
        Uses scale_pos_weight for class imbalance, early stopping, and deeper trees.
        """
        print("Training XGBoost OSDI + Tabular Model (16 expanded features)...")
        X_train = np.array([s['tabular_vector'] for s in train_samples], dtype=np.float32)
        y_train = np.array([s['label'] for s in train_samples], dtype=np.float32)

        X_val = np.array([s['tabular_vector'] for s in val_samples], dtype=np.float32)
        y_val = np.array([s['label'] for s in val_samples], dtype=np.float32)

        # Compute class imbalance ratio for scale_pos_weight
        n_neg = max(1, int(np.sum(y_train == 0)))
        n_pos = max(1, int(np.sum(y_train == 1)))
        scale_pos = float(n_neg / n_pos)
        print(f"  Class balance: {n_neg} Normal / {n_pos} Dry Eye -> scale_pos_weight={scale_pos:.2f}")

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'error', 'auc'],
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.85,
            'colsample_bytree': 0.80,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 1.5,
            'scale_pos_weight': scale_pos,
            'seed': 42
        }

        evals = [(dtrain, 'train'), (dval, 'val')]
        self.booster = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=evals,
            early_stopping_rounds=30,
            verbose_eval=False
        )

        preds_val = self.booster.predict(dval)
        bin_preds = (preds_val >= 0.50).astype(int)
        acc = float(np.mean(bin_preds == y_val))

        best_round = getattr(self.booster, 'best_iteration', 300)
        print(f"XGBoost Tabular Model Trained. Val Accuracy: {acc * 100:.2f}% (best round: {best_round})")
        self.save()
        return acc

    def predict(self, osdi_score, thermal_metrics=None):
        """
        Inference:
          - osdi_score: float (0 to 100)
          - thermal_metrics: optional dict from ThermalFeatureExtractor
        Returns:
          - prob: float probability (0.0 to 1.0)
          - severity_str: "Normal" | "Mild Dry Eye" | "Moderate Dry Eye" | "Severe Dry Eye"
        """
        sev_idx, sev_str = self.get_osdi_severity(osdi_score)

        if thermal_metrics is not None and 'central_cornea_temp' in thermal_metrics:
            re = thermal_metrics.get('re', {})
            le = thermal_metrics.get('le', {})
            cc = float(thermal_metrics.get('central_cornea_temp', self.default_clinical_means['central_cornea_temp']))
            cr = float(thermal_metrics.get('cooling_rate', self.default_clinical_means['cooling_rate_10s']))
            asym = float(thermal_metrics.get('bilateral_asymmetry', self.default_clinical_means['bilateral_asymmetry']))
            cold = float(thermal_metrics.get('tear_breakup_cold_spots', self.default_clinical_means['tear_breakup_cold_spots']))
            re_tmean = float(re.get('t_mean', self.default_clinical_means['re_tmean']))
            re_tstd = float(re.get('t_std', self.default_clinical_means['re_tstd']))
            re_tcc = float(re.get('t_cc', self.default_clinical_means['re_tcc']))
            re_tnc = float(re.get('t_nc', self.default_clinical_means['re_tnc']))
            re_ttc = float(re.get('t_tc', self.default_clinical_means['re_ttc']))
            le_tcc = float(le.get('t_cc', self.default_clinical_means['le_tcc']))
            le_tmean = float(le.get('t_mean', self.default_clinical_means['le_tmean']))
            le_tstd = float(le.get('t_std', self.default_clinical_means['le_tstd']))
            re_nt_grad = re_tnc - re_ttc
            le_cold = float(le.get('cold_spots_ratio', self.default_clinical_means['le_cold_spots']))
        else:
            # Impute standard clinical parameters if running standalone OSDI
            d = self.default_clinical_means
            cc = d['central_cornea_temp']
            cr = d['cooling_rate_10s']
            asym = d['bilateral_asymmetry']
            cold = d['tear_breakup_cold_spots']
            re_tmean = d['re_tmean']
            re_tstd = d['re_tstd']
            re_tcc = d['re_tcc']
            re_tnc = d['re_tnc']
            re_ttc = d['re_ttc']
            le_tcc = d['le_tcc']
            le_tmean = d['le_tmean']
            le_tstd = d['le_tstd']
            re_nt_grad = d['re_nt_gradient']
            le_cold = d['le_cold_spots']

        feature_row = [
            float(osdi_score), float(sev_idx), cc, cr, asym, cold,
            re_tmean, re_tstd, re_tcc, re_tnc, re_ttc,
            le_tcc, le_tmean, le_tstd, re_nt_grad, le_cold
        ]

        if self.booster is not None:
            dmat = xgb.DMatrix(np.array([feature_row], dtype=np.float32), feature_names=self.feature_names)
            prob = float(self.booster.predict(dmat)[0])
        else:
            # Clinical sigmoid baseline if model weights not yet loaded
            # OSDI cutoff 22.0 is clinical boundary for dry eye
            prob = 1.0 / (1.0 + np.exp(-(float(osdi_score) - 22.0) / 7.5))

        return prob, sev_str

    def save(self):
        if self.booster is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.booster.save_model(self.model_path)
            print(f"Saved XGBoost model to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.booster = xgb.Booster()
            self.booster.load_model(self.model_path)
            print(f"Loaded XGBoost model from {self.model_path}")
        else:
            print(f"XGBoost model file not found at {self.model_path}")
