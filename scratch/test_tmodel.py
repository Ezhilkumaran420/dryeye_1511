import json
import numpy as np

# Load thermal_model_data.json
with open('thermal_model_data.json', 'r', encoding='utf-8') as f:
    tmodel = json.load(f)

print("Loaded thermal model:")
print("Weights:", tmodel['weights'])
print("Bias:", tmodel['bias'])
print("Features:", tmodel['feature_labels'])
print("Sample Gallery count:", len(tmodel['sample_gallery']))

for sample in tmodel['sample_gallery'][:3]:
    print("Sample:", sample['id'], sample['re_diagnosis'], "RE CC:", sample['re_cc'], "LE CC:", sample['le_cc'])
