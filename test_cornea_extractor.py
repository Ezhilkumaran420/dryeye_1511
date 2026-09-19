import os, glob
from PIL import Image
import numpy as np

def locate_cornea_and_extract_temps(img_path):
    im = Image.open(img_path)
    arr = np.array(im)
    
    # Scale bar calibration
    scale_y = np.arange(29, 209)
    scale_colors = arr[scale_y, 308, :].astype(float)
    t_max = 37.0
    t_min = 27.0
    scale_temps = t_max - (scale_y - 29) / (208 - 29) * (t_max - t_min)
    
    def analyze_eye(sub_crop):
        h, w, _ = sub_crop.shape
        flat = sub_crop.reshape(-1, 3).astype(float)
        diff = np.sum((flat[:, None, :] - scale_colors[None, :, :])**2, axis=-1)
        best = np.argmin(diff, axis=-1)
        temps = scale_temps[best].reshape(h, w)
        
        # Cornea is in central 60% of crop
        cy, cx = h // 2, w // 2
        cornea_patch = temps[max(0, cy-15):min(h, cy+15), max(0, cx-20):min(w, cx+20)]
        
        return {
            'mean': float(np.mean(cornea_patch)),
            'min': float(np.percentile(cornea_patch, 10)),
            'max': float(np.percentile(cornea_patch, 90)),
            'std': float(np.std(cornea_patch))
        }
        
    re_info = analyze_eye(arr[110:170, 50:140])
    le_info = analyze_eye(arr[110:170, 175:265])
    return re_info, le_info

for sno in [1, 8, 14, 45, 52]:
    p0 = rf'D:\AI+ML\{sno} - 0S.jpg'
    if os.path.exists(p0):
        re, le = locate_cornea_and_extract_temps(p0)
        print(f"Subject {sno:2d}: RE mean={re['mean']:.2f}, min={re['min']:.2f}, std={re['std']:.2f} | LE mean={le['mean']:.2f}, min={le['min']:.2f}, std={le['std']:.2f}")
