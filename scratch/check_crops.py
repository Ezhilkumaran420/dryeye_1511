from PIL import Image
import numpy as np
import glob, os

# Let's inspect where the brightest/warmest corneal center is in 10 subjects
for sno in [1, 2, 4, 8, 10, 11, 15, 30, 45, 60]:
    p = f"D:/AI+ML/{sno} - 0S.jpg"
    if not os.path.exists(p):
        continue
    im = Image.open(p)
    arr = np.array(im)
    
    # Scale bar calibration
    scale_y = np.arange(29, 209)
    scale_colors = arr[scale_y, 308, :].astype(float)
    t_max, t_min = 37.0, 27.0
    scale_temps = t_max - (scale_y - 29) / (208 - 29) * (t_max - t_min)
    
    # Convert whole image (320x240) to temperatures
    flat = arr.reshape(-1, 3).astype(float)
    diff = np.sum((flat[:, None, :] - scale_colors[None, :, :])**2, axis=-1)
    best = np.argmin(diff, axis=-1)
    temps = scale_temps[best].reshape(240, 320)
    
    # Left eye region (in image: right side x in 160..280, y in 80..180)
    # Right eye region (in image: left side x in 40..160, y in 80..180)
    re_crop = temps[90:170, 40:150]
    le_crop = temps[90:170, 160:270]
    
    # The cornea is typically an elliptical zone where temperatures are between 33 and 36.5
    # The nose / canthus is warmer (> 36.5)
    re_min_y, re_min_x = np.unravel_index(np.argmin(re_crop), re_crop.shape)
    le_min_y, le_min_x = np.unravel_index(np.argmin(le_crop), le_crop.shape)
    
    re_mean = np.mean(re_crop)
    le_mean = np.mean(le_crop)
    
    print(f"SNo {sno:2d}: RE crop mean={re_mean:.2f} (min={np.min(re_crop):.2f} at {re_min_x+40},{re_min_y+90}) | LE crop mean={le_mean:.2f} (min={np.min(le_crop):.2f} at {le_min_x+160},{le_min_y+90})")
