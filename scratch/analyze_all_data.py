import openpyxl
import os, glob, re
from PIL import Image
import numpy as np

# Load both Excels
eye_path = r"C:\Users\ezhil\Downloads\Eye classification.xlsx"
npd_path = r"C:\Users\ezhil\Downloads\N & PD Eye.xlsx"
img_dir = r"D:\AI+ML"

wb_eye = openpyxl.load_workbook(eye_path, data_only=True)
wb_npd = openpyxl.load_workbook(npd_path, data_only=True)

# Parse RE and LE
re_rows = list(wb_eye['RE'].iter_rows(values_only=True))
le_rows = list(wb_eye['LE'].iter_rows(values_only=True))
npd_rows = list(wb_npd['OSDI_Final'].iter_rows(values_only=True))

re_head = re_rows[0]
le_head = le_rows[0]

print("RE headers:", re_head)
print("LE headers:", le_head)

patients = {}
for r in re_rows[1:]:
    sno = int(r[0])
    lbl = str(r[-1]).strip()
    is_dry = 1 if 'dry' in lbl.lower() else 0
    patients[sno] = {
        'sno': sno,
        'osdi': float(r[1] or 0),
        'age': r[2],
        'gender': r[3],
        're_cr10': float(r[4] or 0),
        're_cr7': float(r[5] or 0),
        're_nc': float(r[6] or 0),
        're_tc': float(r[7] or 0),
        're_cc': float(r[8] or 0),
        're_nl': float(r[9] or 0),
        're_tl': float(r[10] or 0),
        're_t0': float(r[11] or 0),
        're_t10': float(r[12] or 0),
        're_most': float(r[13] or 0),
        're_label': is_dry,
        're_label_str': 'Possible Dry Eye' if is_dry else 'Normal'
    }

for r in le_rows[1:]:
    sno = int(r[0])
    lbl = str(r[-1]).strip()
    is_dry = 1 if 'dry' in lbl.lower() else 0
    if sno in patients:
        patients[sno].update({
            'le_cr10': float(r[2] or 0),
            'le_cr7': float(r[3] or 0),
            'le_nc': float(r[4] or 0),
            'le_tc': float(r[5] or 0),
            'le_cc': float(r[6] or 0),
            'le_nl': float(r[7] or 0),
            'le_tl': float(r[8] or 0),
            'le_t0': float(r[9] or 0),
            'le_t10': float(r[10] or 0),
            'le_most': float(r[11] or 0),
            'le_label': is_dry,
            'le_label_str': 'Possible Dry Eye' if is_dry else 'Normal'
        })

print(f"Loaded {len(patients)} complete patients from Eye classification.xlsx")

# Check N & PD Eye.xlsx
npd_dict = {}
for r in npd_rows[1:]:
    sno = int(r[0])
    re_l = str(r[1]).strip()
    osdi = float(r[2] or 0)
    le_l = str(r[3]).strip()
    npd_dict[sno] = {
        're_label': 'Possible Dry Eye' if 'dry' in re_l.lower() else 'Normal',
        'le_label': 'Possible Dry Eye' if 'dry' in le_l.lower() else 'Normal',
        'osdi': osdi
    }

print(f"Loaded {len(npd_dict)} patients from N & PD Eye.xlsx")

# Check consistency
mismatches = 0
for sno, p in patients.items():
    if sno in npd_dict:
        np = npd_dict[sno]
        if p['re_label_str'] != np['re_label'] or p['le_label_str'] != np['le_label']:
            mismatches += 1
            print(f"Mismatch for SNo {sno}: Eye=RE:{p['re_label_str']}, LE:{p['le_label_str']} vs NPD=RE:{np['re_label']}, LE:{np['le_label']}")

print(f"Total mismatches between two Excels: {mismatches}")
