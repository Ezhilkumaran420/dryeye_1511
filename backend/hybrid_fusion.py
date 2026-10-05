import numpy as np


class HybridFusionEngine:
    """
    Combines:
      - Thermal Image Analysis (CNN) probability
      - OSDI Score (Tabular / Clinical Model) probability
    Supports both fixed-weight and learned stacking fusion.
    Generates:
      - Diagnosis ("Normal" | "Dry Eye")
      - Confidence (0.0 to 100.0%)
      - Severity ("None" | "Mild" | "Moderate" | "Severe")
      - Temperature Status ("Normal" | "Abnormal")
      - Clinical Recommendation
    """
    def __init__(self, image_weight=0.80, osdi_weight=0.20, decision_threshold=0.50):
        self.image_weight = image_weight
        self.osdi_weight = osdi_weight
        self.threshold = decision_threshold
        # Learned stacking weights (trained via fit_stacking)
        self.stacking_weights = None
        self.stacking_bias = None

    def fit_stacking(self, image_probs, osdi_probs, thermal_signs, y_true):
        """
        Learns optimal fusion weights via L2-regularized logistic regression stacking.
        Inputs: arrays of probabilities from image model, OSDI model, and thermal sign scores.
        Trains a small logistic regression: sigmoid(w1*p_img + w2*p_osdi + w3*t_sign + b)
        """
        image_probs = np.array(image_probs, dtype=np.float64)
        osdi_probs = np.array(osdi_probs, dtype=np.float64)
        thermal_signs = np.array(thermal_signs, dtype=np.float64)
        y = np.array(y_true, dtype=np.float64)

        # Feature matrix for stacking: [p_img, p_osdi, therm_sign]
        X = np.column_stack([image_probs, osdi_probs, thermal_signs])

        # Normalize features for stable optimization
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std[x_std < 1e-8] = 1.0
        X_norm = (X - x_mean) / x_std

        # L2-regularized logistic regression (gradient descent)
        N, D = X_norm.shape
        w = np.zeros(D)
        b = 0.0
        lr = 0.5
        reg = 0.01

        for _ in range(500):
            z = X_norm @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
            grad_w = X_norm.T @ (p - y) / N + reg * w
            grad_b = np.mean(p - y)
            w -= lr * grad_w
            b -= lr * grad_b

        # Convert weights back to original (unnormalized) input space
        w_orig = w / x_std
        b_orig = b - np.sum(w * x_mean / x_std)

        self.stacking_weights = w_orig.tolist()
        self.stacking_bias = float(b_orig)

        # Report stacking accuracy
        z_final = X @ np.array(self.stacking_weights) + self.stacking_bias
        p_final = 1.0 / (1.0 + np.exp(-np.clip(z_final, -15.0, 15.0)))
        acc = np.mean((p_final >= 0.5).astype(int) == y)
        print(f"Learned Stacking Fusion: weights={[round(w, 4) for w in self.stacking_weights]}, "
              f"bias={self.stacking_bias:.4f}, train_acc={acc * 100:.2f}%")

    def _fuse_score(self, image_prob, osdi_prob, thermal_sign=0.0):
        """
        Computes fused probability using learned stacking or fixed weights.
        """
        p_img = float(max(0.0, min(1.0, image_prob)))
        p_osdi = float(max(0.0, min(1.0, osdi_prob)))
        t_sign = float(max(0.0, min(1.0, thermal_sign)))

        if self.stacking_weights is not None:
            z = (self.stacking_weights[0] * p_img +
                 self.stacking_weights[1] * p_osdi +
                 self.stacking_weights[2] * t_sign +
                 self.stacking_bias)
            return float(1.0 / (1.0 + np.exp(-max(-15.0, min(15.0, z)))))
        else:
            # Fallback to fixed-weight linear combination
            return float(self.image_weight * p_img + self.osdi_weight * p_osdi)

    def compute_fusion(self, image_prob, osdi_prob, osdi_score, thermal_metrics=None, thermal_sign=0.0):
        """
        Calculates hybrid fusion prediction and structured clinical report.
        """
        final_score = self._fuse_score(image_prob, osdi_prob, thermal_sign)
        is_dry = final_score >= self.threshold
        diagnosis = "Dry Eye" if is_dry else "Normal"

        # Confidence percentage (distance from uncertainty 0.5 scaled to 50-100%)
        if is_dry:
            conf = final_score * 100.0
        else:
            conf = (1.0 - final_score) * 100.0
        confidence = round(max(50.0, min(99.9, conf)), 1)

        # Severity determination
        if not is_dry:
            severity = "None"
        else:
            if osdi_score is not None and osdi_score >= 33.0 or final_score >= 0.82:
                severity = "Severe"
            elif osdi_score is not None and osdi_score >= 23.0 or final_score >= 0.65:
                severity = "Moderate"
            else:
                severity = "Mild"

        # Temperature status validation
        temp_status = "Normal"
        if thermal_metrics:
            temp_status = thermal_metrics.get('temperature_status', 'Normal')
            cc = thermal_metrics.get('central_cornea_temp', 35.0)
            asym = thermal_metrics.get('bilateral_asymmetry', 0.2)
            if cc < 34.0 or asym >= 0.50 or is_dry:
                # If dry eye with thermal anomaly
                if cc < 34.0 or asym >= 0.45 or thermal_metrics.get('tear_breakup_cold_spots', 0) > 0.12:
                    temp_status = "Abnormal"

        # Clinical Recommendation
        recommendation = self._generate_recommendation(diagnosis, severity, osdi_score, temp_status, thermal_metrics)

        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "image_prediction": round(float(max(0.0, min(1.0, image_prob))), 4),
            "osdi_prediction": round(float(max(0.0, min(1.0, osdi_prob))), 4),
            "final_score": round(final_score, 4),
            "severity": severity,
            "temperature_status": temp_status,
            "recommendation": recommendation,
            "weights": {
                "thermal_image": self.image_weight,
                "osdi_tabular": self.osdi_weight,
                "stacking_learned": self.stacking_weights is not None
            }
        }

    def _generate_recommendation(self, diagnosis, severity, osdi_score, temp_status, metrics):
        if diagnosis == "Normal":
            if temp_status == "Abnormal":
                return ("Ocular surface is within normal healthy limits, but slight thermal variation detected. "
                        "Maintain adequate blink frequency during screen work and maintain hydration.")
            return ("Ocular surface temperature and tear film stability are healthy. "
                    "Routine annual examination recommended.")

        # Dry Eye Recommendations
        lines = []
        if severity == "Mild":
            lines.append("Possiblity Of Mild Dry Eye Detected.")
            lines.append("Begin preservative-free artificial tear drops (3-4 times daily).")
            lines.append("Practice the 20-20-20 rule during digital device use and apply warm compresses in the evening.")
        elif severity == "Moderate":
            lines.append("Possiblity Of Moderate Dry Eye Detected with noticeable tear film disruption.")
            lines.append("Initiate lipid-based lubricating eye drops 4-6 times daily.")
            lines.append("Perform nightly eyelid hygiene with warm compresses (10 minutes) and consider nighttime lubricating ointment.")
            lines.append("Schedule an in-person slit lamp and tear breakup time (TBUT) examination.")
        else: # Severe
            lines.append("Possiblity Of Severe Dry Eye Detected with critical thermal instability and symptom burden.")
            lines.append("Urgent ophthalmological evaluation recommended for punctal occlusion or prescription anti-inflammatory therapy (e.g., Cyclosporine/Lifitegrast).")
            lines.append("Use intensive lubricating gels, moisture chamber glasses, and avoid direct air currents or dry environments.")

        if temp_status == "Abnormal":
            lines.append("Significant corneal temperature asymmetry or cold spots indicate rapid localized tear film breakup.")

        return " ".join(lines)
