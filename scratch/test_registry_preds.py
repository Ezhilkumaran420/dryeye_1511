import json
import numpy as np

with open("scratch/clinical_models_and_registry.json") as f:
    data = json.load(f)

reg = data['clinical_registry']
m = data['model_thermal_only']
w = np.array(m['weights'])
b = m['bias']
mean = np.array(m['mean'])
std = np.array(m['std'])
thresh = m['threshold']

print(f"Testing predictions on all 78 subjects:")
correct = 0
for sno_str, s in reg.items():
    sno = int(sno_str)
    for eye in ['re', 'le']:
        ed = s[eye]
        feat = np.array([
            ed['cr10'], ed['cr7'], ed['nc'], ed['tc'], ed['cc'], ed['nl'], ed['tl'], ed['t0'], ed['t10'], ed['most']
        ])
        norm = (feat - mean) / std
        z = np.dot(norm, w) + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
        pred = 1 if p >= thresh else 0
        actual = ed['is_dry']
        if pred == actual:
            correct += 1
        else:
            print(f"Subject {sno} {eye.upper()}: Pred={'Dry' if pred else 'Normal'} ({p*100:.1f}%), Actual={'Dry' if actual else 'Normal'}")

print(f"Total correct: {correct}/156 ({correct/156*100:.2f}%)")
