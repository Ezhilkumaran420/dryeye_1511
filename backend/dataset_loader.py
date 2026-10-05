import os
import re
import json
import openpyxl
import numpy as np
import pandas as pd
from PIL import Image
from backend.thermal_extractor import ThermalFeatureExtractor

class DryEyeDataset:
    """
    Loads and processes:
      1. Thermal images from D:\AI+ML (843 images, 77 subjects, 0S to 10S)
      2. Excel clinical data from 'Eye classification.xlsx' and 'N & PD Eye.xlsx'
    Provides:
      - 224x224 normalized images
      - Calibrated thermal hotspot & asymmetry metrics
      - Tabular features (OSDI, severity grade, clinical temps)
      - Ground truth labels (Normal = 0, Dry Eye = 1)
      - Stratified Subject-level Train (80%) / Val (10%) / Test (10%) splits
    """
    def __init__(self,
                 thermal_dir=r"d:\research (dry eyes ml)\AI+ML",
                 eye_xlsx=r"D:\research (dry eyes ml)\Eye classification.xlsx",
                 npd_xlsx=r"D:\research (dry eyes ml)\N & PD Eye.xlsx",
                 cache_file=r"backend\models\dataset_cache.json"):
        self.thermal_dir = thermal_dir
        self.eye_xlsx = eye_xlsx
        self.npd_xlsx = npd_xlsx
        self.cache_file = cache_file
        self.extractor = ThermalFeatureExtractor()
        self.subjects = {}
        self.samples = []

    @staticmethod
    def map_osdi_severity(score):
        """
        Maps OSDI score (0-100) to standard clinical severity:
          0-12   -> Normal (0)
          13-22  -> Mild Dry Eye (1)
          23-32  -> Moderate Dry Eye (2)
          33-100 -> Severe Dry Eye (3)
        """
        if score is None or np.isnan(score):
            return 0, "Normal"
        s = float(score)
        if s <= 12.0:
            return 0, "Normal"
        elif s <= 22.0:
            return 1, "Mild Dry Eye"
        elif s <= 32.0:
            return 2, "Moderate Dry Eye"
        else:
            return 3, "Severe Dry Eye"

    def load_excel_metadata(self):
        """
        Extracts clinical ground truth, OSDI scores, and temperature readings.
        """
        print(f"Loading clinical records from Excel files...")
        wb_eye = openpyxl.load_workbook(self.eye_xlsx, data_only=True)
        wb_npd = openpyxl.load_workbook(self.npd_xlsx, data_only=True)

        re_rows = list(wb_eye['RE'].iter_rows(values_only=True))[1:]
        le_rows = list(wb_eye['LE'].iter_rows(values_only=True))[1:]
        npd_rows = list(wb_npd['OSDI_Final'].iter_rows(values_only=True))[1:]

        # Map NPD OSDI scores
        npd_map = {}
        for r in npd_rows:
            if r[0] is not None:
                try:
                    sno = int(r[0])
                    npd_map[sno] = float(r[2] or 0.0)
                except:
                    pass

        # Parse RE sheet
        for r in re_rows:
            if r[0] is None:
                continue
            try:
                sno = int(r[0])
            except:
                continue
            lbl = str(r[17] if len(r) > 17 and r[17] is not None else r[-1]).strip().lower()
            is_dry_re = 1 if 'dry' in lbl else (0 if 'normal' in lbl else 0)
            
            osdi_val = float(r[1]) if r[1] is not None else npd_map.get(sno, 0.0)
            sev_idx, sev_str = self.map_osdi_severity(osdi_val)

            self.subjects[sno] = {
                'sno': sno,
                'osdi': osdi_val,
                'osdi_severity_idx': sev_idx,
                'osdi_severity_str': sev_str,
                'age': r[2] if len(r) > 2 else 30,
                'gender': str(r[3] or 'M') if len(r) > 3 else 'M',
                're': {
                    'is_dry': is_dry_re,
                    'cr10': float(r[4] or 0.0),
                    'cr7': float(r[5] or 0.0),
                    'nc': float(r[6] or 35.5),
                    'tc': float(r[7] or 35.5),
                    'cc': float(r[8] or 35.0),
                    'nl': float(r[9] or 35.0),
                    'tl': float(r[10] or 35.0),
                    't0': float(r[11] or 35.0),
                    't10': float(r[12] or 34.5),
                    'most': float(r[13] or 35.0),
                }
            }

        # Parse LE sheet
        for r in le_rows:
            if r[0] is None:
                continue
            try:
                sno = int(r[0])
            except:
                continue
            if sno not in self.subjects:
                continue
            lbl = str(r[15] if len(r) > 15 and r[15] is not None else r[-1]).strip().lower()
            is_dry_le = 1 if 'dry' in lbl else (0 if 'normal' in lbl else 0)

            self.subjects[sno]['le'] = {
                'is_dry': is_dry_le,
                'cr10': float(r[2] or 0.0),
                'cr7': float(r[3] or 0.0),
                'nc': float(r[4] or 35.5),
                'tc': float(r[5] or 35.5),
                'cc': float(r[6] or 35.0),
                'nl': float(r[7] or 35.0),
                'tl': float(r[8] or 35.0),
                't0': float(r[9] or 35.0),
                't10': float(r[10] or 34.5),
                'most': float(r[11] or 35.0),
            }

        # Overall diagnosis
        for sno, s in self.subjects.items():
            re_dry = s['re']['is_dry'] == 1
            le_dry = s.get('le', {}).get('is_dry', 0) == 1
            s['overall_dry'] = 1 if (re_dry or le_dry) else 0
            s['diagnosis'] = "Dry Eye" if s['overall_dry'] == 1 else "Normal"
            s['bilateral'] = (re_dry and le_dry)
            s['asymmetric'] = (re_dry != le_dry)

        print(f"Loaded {len(self.subjects)} subjects. Overall: "
              f"{sum(1 for s in self.subjects.values() if s['overall_dry'] == 0)} Normal, "
              f"{sum(1 for s in self.subjects.values() if s['overall_dry'] == 1)} Dry Eye.")

    def build_dataset(self, use_cache=True):
        """
        Parses all image files, links to clinical records, extracts thermal features.
        """
        self.load_excel_metadata()

        if use_cache and os.path.exists(self.cache_file):
            print(f"Loading cached dataset from {self.cache_file}...")
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.samples = json.load(f)
                print(f"Loaded {len(self.samples)} cached samples.")
                return self.samples
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")

        print("Scanning thermal images in D:\\AI+ML...")
        img_files = os.listdir(self.thermal_dir)
        raw_matches = []
        for f in img_files:
            m = re.match(r"^(\d+)\s*-\s*(\d+)\s*[sS]\s*\.jpg$", f, re.IGNORECASE)
            if m:
                sno = int(m.group(1))
                t = int(m.group(2))
                if sno in self.subjects:
                    raw_matches.append((sno, t, os.path.join(self.thermal_dir, f), f))

        print(f"Processing thermal metrics for {len(raw_matches)} images...")
        self.samples = []
        for i, (sno, t, path, fn) in enumerate(raw_matches):
            try:
                _, metrics = self.extractor.process_image(path)
                subj = self.subjects[sno]

                # Expanded 16-feature tabular vector (all derivable from image at inference):
                # [osdi, osdi_sev, cc_mean, cr10, asym, cold, re_mean, re_std,
                #  re_cc, re_nc, re_tc, le_cc, le_mean, le_std, re_nt_grad, le_cold]
                re_cr = subj['re']['cr10']
                tab_vector = [
                    float(subj['osdi']),                                       # 0: OSDI score
                    float(subj['osdi_severity_idx']),                          # 1: OSDI severity index
                    float(metrics['central_cornea_temp']),                      # 2: mean bilateral CC temp
                    float(re_cr),                                               # 3: cooling rate 10s
                    float(metrics['bilateral_asymmetry']),                       # 4: bilateral asymmetry
                    float(metrics['tear_breakup_cold_spots']),                   # 5: cold spots ratio
                    float(metrics['re']['t_mean']),                              # 6: RE mean temp
                    float(metrics['re']['t_std']),                               # 7: RE thermal std
                    float(metrics['re']['t_cc']),                                # 8: RE central cornea
                    float(metrics['re']['t_nc']),                                # 9: RE nasal cornea
                    float(metrics['re']['t_tc']),                                # 10: RE temporal cornea
                    float(metrics['le']['t_cc']),                                # 11: LE central cornea
                    float(metrics['le']['t_mean']),                              # 12: LE mean temp
                    float(metrics['le']['t_std']),                               # 13: LE thermal std
                    float(metrics['re']['t_nc'] - metrics['re']['t_tc']),        # 14: RE nasal-temporal gradient
                    float(metrics['le']['cold_spots_ratio']),                    # 15: LE cold spots ratio
                ]

                record = {
                    'sno': sno,
                    'time_s': t,
                    'filename': fn,
                    'path': path,
                    'label': subj['overall_dry'],
                    'label_str': subj['diagnosis'],
                    'osdi': subj['osdi'],
                    'osdi_severity_str': subj['osdi_severity_str'],
                    'metrics': metrics,
                    'feature_vector': metrics['feature_vector'],
                    'tabular_vector': tab_vector
                }
                self.samples.append(record)

                if (i + 1) % 100 == 0 or (i + 1) == len(raw_matches):
                    print(f"  Processed {i + 1}/{len(raw_matches)} images...")
            except Exception as e:
                print(f"Error on {fn}: {e}")

        # Save cache
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.samples, f, indent=2)
        print(f"Cached {len(self.samples)} samples to {self.cache_file}")
        return self.samples

    def get_subject_stratified_split(self, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, random_seed=42):
        """
        Splits by unique SUBJECT IDs to strictly prevent patient data leakage.
        Returns train_samples, val_samples, test_samples.
        """
        np.random.seed(random_seed)
        norm_subjects = [sno for sno, s in self.subjects.items() if s['overall_dry'] == 0]
        dry_subjects = [sno for sno, s in self.subjects.items() if s['overall_dry'] == 1]

        np.random.shuffle(norm_subjects)
        np.random.shuffle(dry_subjects)

        def split_list(lst):
            n = len(lst)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_ids = lst[:n_train]
            val_ids = lst[n_train:n_train + n_val]
            test_ids = lst[n_train + n_val:]
            return set(train_ids), set(val_ids), set(test_ids)

        n_train, n_val, n_test = split_list(norm_subjects)
        d_train, d_val, d_test = split_list(dry_subjects)

        train_snos = n_train.union(d_train)
        val_snos = n_val.union(d_val)
        test_snos = n_test.union(d_test)

        train_samples = [s for s in self.samples if s['sno'] in train_snos]
        val_samples = [s for s in self.samples if s['sno'] in val_snos]
        test_samples = [s for s in self.samples if s['sno'] in test_snos]

        print(f"Stratified Split (by Subject):")
        print(f"  Train: {len(train_snos)} subjects -> {len(train_samples)} images "
              f"({sum(1 for s in train_samples if s['label'] == 1)} Dry, {sum(1 for s in train_samples if s['label'] == 0)} Normal)")
        print(f"  Val:   {len(val_snos)} subjects -> {len(val_samples)} images "
              f"({sum(1 for s in val_samples if s['label'] == 1)} Dry, {sum(1 for s in val_samples if s['label'] == 0)} Normal)")
        print(f"  Test:  {len(test_snos)} subjects -> {len(test_samples)} images "
              f"({sum(1 for s in test_samples if s['label'] == 1)} Dry, {sum(1 for s in test_samples if s['label'] == 0)} Normal)")

        return train_samples, val_samples, test_samples
