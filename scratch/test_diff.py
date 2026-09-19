import openpyxl
import numpy as np

wb = openpyxl.load_workbook(r"C:\Users\ezhil\Downloads\Eye classification.xlsx", data_only=True)

for eye in ['RE', 'LE']:
    print(f"\n=== {eye} ===")
    ws = wb[eye]
    rows = list(ws.iter_rows(values_only=True))[1:]
    
    col_map = {
        'OSDI': 1,
        'CR10': 4 if eye == 'RE' else 2,
        'NC': 6 if eye == 'RE' else 4,
        'TC': 7 if eye == 'RE' else 5,
        'CC': 8 if eye == 'RE' else 6,
        'NL': 9 if eye == 'RE' else 7,
        'TL': 10 if eye == 'RE' else 8,
        'MOST': 13 if eye == 'RE' else 11
    }
    
    norm_rows = [r for r in rows if 'normal' in str(r[-1]).lower()]
    dry_rows = [r for r in rows if 'dry' in str(r[-1]).lower()]
    print(f"Counts: Normal={len(norm_rows)}, Dry={len(dry_rows)}")
    
    for name, idx in col_map.items():
        v_norm = [float(r[idx] or 0) for r in norm_rows]
        v_dry = [float(r[idx] or 0) for r in dry_rows]
        print(f"  {name:6s}: Normal mean={np.mean(v_norm):.2f} (std={np.std(v_norm):.2f}) | Dry mean={np.mean(v_dry):.2f} (std={np.std(v_dry):.2f})")
