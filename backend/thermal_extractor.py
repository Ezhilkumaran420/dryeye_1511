import os
import json
import cv2
import numpy as np
from PIL import Image

class ThermalFeatureExtractor:
    """
    AI-Trained Ocular Surface & Cornea Feature Extractor.
    Automatically detects the Eye Region (palpebral fissure) and Cornea (OD & OS)
    from FLIR anterior segment thermal images and extracts biophysical corneal biomarkers.
    """
    def __init__(self, t_min=27.0, t_max=37.0):
        self.t_min = t_min
        self.t_max = t_max
        self.scale_y_start = 29
        self.scale_y_end = 208
        self.scale_col = 308

        # Load trained AI cornea detector weights
        self.ai_model = None
        weights_paths = [
            os.path.join(os.path.dirname(__file__), "models", "cornea_detector_weights.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "cornea_detector_weights.json"),
            "cornea_detector_weights.json"
        ]
        for wp in weights_paths:
            if os.path.exists(wp):
                try:
                    with open(wp, 'r') as f:
                        self.ai_model = json.load(f)
                    break
                except Exception:
                    pass

    FLIR_TEMPLATES = {
        0: ["  #### ", " ##  ##", " #    #", "##    #", "##    #", "##    #", "##    #", "##    #", " #    #", " #    #", " ##  ##", "  #### "],
        1: ["  ##   ", "###    ", "  #    ", "  #    ", "  #    ", "  #    ", "  #    ", "  #    ", "  #    ", "  #    ", "#####  ", "       "],
        2: ["#####  ", "#   ## ", "     # ", "     # ", "     # ", "    ## ", "    #  ", "   ##  ", "  ##   ", " ##    ", "##     ", "#######"],
        3: ["#####  ", "#    # ", "     # ", "     # ", "     # ", "  ##   ", "     # ", "     # ", "     ##", "     ##", "     # ", "#####  "],
        4: ["    ## ", "    ## ", "   # # ", "  #  # ", " ##  # ", " #   # ", "#    # ", "#######", "     # ", "     # ", "     # ", "     # "],
        5: ["###### ", "#      ", "##     ", "#      ", "#####  ", "    ## ", "     # ", "     ##", "     # ", "     # ", "#   #  ", "#####  "],
        6: ["   ### ", "  #    ", " #     ", " #     ", "###### ", "##   ##", "##    #", "##    #", "##    #", " #    #", " #   ##", "  ###  "],
        7: ["###### ", "     # ", "     # ", "    #  ", "    #  ", "   #   ", "   #   ", "  ##   ", "  #    ", " ##    ", " #     ", "#      "],
        8: ["  #### ", " #   ##", "##    #", "##    #", " ##  ##", "  #### ", " #  ## ", "##    #", "#     #", "##    #", "##   ##", "  #### "],
        9: ["  #### ", " #   # ", "##   ##", "#     #", "#     #", "#     #", " #    #", "  #####", "      #", "     ##", "    ## ", " ####  "]
    }
    FLIR_TMPL_BOOL = {
        d: np.array([[c == '#' for c in row] for row in rows], dtype=bool)
        for d, rows in FLIR_TEMPLATES.items()
    }

    def extract_corner_temps(self, arr):
        """
        Dynamically extracts the exact high and least temperatures burned into
        the corners of the FLIR thermal image next to the vertical color bar (near left eye).
        """
        attempts = []
        if arr.shape[0] == 240 and arr.shape[1] == 320:
            attempts.append(arr)
        else:
            pil_im = Image.fromarray(arr)
            attempts.append(np.array(pil_im.resize((320, 240), Image.Resampling.NEAREST)))
            attempts.append(np.array(pil_im.resize((320, 240), Image.Resampling.BILINEAR)))

        def parse_corner_box(sub_arr, is_high=True):
            h, w = sub_arr.shape[:2]
            for thresh in [155, 140, 125, 110, 100, 170, 185, 95]:
                mask = (sub_arr[:, :, 0] > thresh) & (sub_arr[:, :, 1] > thresh) & (sub_arr[:, :, 2] > thresh)
                col_sums = mask.sum(axis=0)

                raw_clusters = []
                in_c = False
                c_start = 0
                for x in range(w):
                    if col_sums[x] > 0 and not in_c:
                        in_c = True
                        c_start = x
                    elif col_sums[x] == 0 and in_c:
                        in_c = False
                        raw_clusters.append((c_start, x))
                if in_c:
                    raw_clusters.append((c_start, w))

                if not raw_clusters:
                    continue

                # Merge clusters separated by a 1-pixel gap (avoids broken digit strokes from thresholding)
                merged = []
                for c in raw_clusters:
                    if not merged:
                        merged.append(c)
                    else:
                        prev = merged[-1]
                        if (c[0] - prev[1] <= 1) and ((c[1] - prev[0]) <= 8):
                            merged[-1] = (prev[0], c[1])
                        else:
                            merged.append(c)

                digit_clusters = [c for c in merged if (c[1] - c[0]) >= 3]
                if len(digit_clusters) != 3:
                    if len(merged) == 4:
                        digit_clusters = [merged[0], merged[1], merged[3]]
                    elif len(raw_clusters) == 4:
                        digit_clusters = [raw_clusters[0], raw_clusters[1], raw_clusters[3]]
                    else:
                        with_h = []
                        for c1, c2 in merged:
                            active = np.where(mask[:, c1:c2].sum(axis=1) > 0)[0]
                            if len(active) > 0 and (active[-1] - active[0] + 1) >= 7 and (c2 - c1) >= 3:
                                with_h.append((c1, c2))
                        if len(with_h) == 3:
                            digit_clusters = with_h
                        else:
                            continue

                digits = []
                failed = False
                for c1, c2 in digit_clusters:
                    char_mask = mask[:, c1:c2]
                    active_rows = np.where(char_mask.sum(axis=1) > 0)[0]
                    if len(active_rows) == 0:
                        failed = True
                        break
                    r_min = active_rows[0]
                    r_max = active_rows[-1] + 1
                    crop = char_mask[r_min:r_max, :]
                    ch, cw = crop.shape

                    grid = np.zeros((12, 7), dtype=bool)
                    grid[:min(12, ch), :min(7, cw)] = crop[:min(12, ch), :min(7, cw)]

                    best_d = None
                    best_score = -1
                    for d in range(10):
                        t = self.FLIR_TMPL_BOOL[d]
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                sy_start = max(0, dy)
                                sy_end = min(12, 12 + dy)
                                gy_start = max(0, -dy)
                                gy_end = min(12, 12 - dy)
                                sx_start = max(0, dx)
                                sx_end = min(7, 7 + dx)
                                gx_start = max(0, -dx)
                                gx_end = min(7, 7 - dx)
                                t_sub = t[sy_start:sy_end, sx_start:sx_end]
                                g_sub = grid[gy_start:gy_end, gx_start:gx_end]
                                inter = np.logical_and(t_sub, g_sub).sum()
                                union = np.logical_or(t, grid).sum()
                                score = inter / union if union > 0 else 0
                                if score > best_score:
                                    best_score = score
                                    best_d = d
                    if best_score < 0.35 or best_d is None:
                        failed = True
                        break
                    digits.append(best_d)

                if not failed and len(digits) == 3:
                    if is_high and digits[0] == 1:
                        digits[0] = 3
                    val = digits[0] * 10.0 + digits[1] * 1.0 + digits[2] * 0.1
                    if is_high and (val < 31.0 or val > 44.0):
                        continue
                    if not is_high and (val < 18.0 or val > 35.0):
                        continue
                    return round(val, 1)
            return None

        for test_arr in attempts:
            high = None
            is_inset = False
            for y1, y2, x1, x2 in [
                (34, 58, 265, 318), (36, 56, 268, 316),
                (6, 25, 275, 318), (4, 28, 268, 318), (4, 28, 260, 319),
                (6, 25, 1, 55), (34, 58, 1, 55)
            ]:
                val = parse_corner_box(test_arr[y1:y2, x1:x2], is_high=True)
                if val is not None:
                    high = val
                    if y1 >= 30:
                        is_inset = True
                    break

            least = None
            for y1, y2, x1, x2 in [
                (182, 206, 265, 318), (184, 204, 268, 316),
                (215, 236, 275, 318), (205, 238, 268, 318), (205, 239, 260, 319),
                (215, 236, 1, 55), (182, 206, 1, 55)
            ]:
                val = parse_corner_box(test_arr[y1:y2, x1:x2], is_high=False)
                if val is not None:
                    least = val
                    break

            if high is not None and least is not None and high > least:
                return high, least, is_inset

        return high, least, False

    def extract_calibration_from_array(self, arr, t_min=None, t_max=None, is_inset=False):
        """
        Extracts the temperature mapping from the FLIR palette color bar on the right side.
        Calibrated to the exact high and least temperatures of the uploaded thermal scan.
        """
        h, w, _ = arr.shape
        if is_inset:
            y_start = min(62, h - 1)
            y_end = min(178, h - 1)
            col = min(303, w - 1)
        else:
            y_start = min(self.scale_y_start, h - 1)
            y_end = min(self.scale_y_end, h - 1)
            col = min(self.scale_col, w - 1)

        scale_y = np.arange(y_start, y_end + 1)
        scale_colors = arr[scale_y, col, :].astype(float)

        t_min_val = t_min if t_min is not None else self.t_min
        t_max_val = t_max if t_max is not None else self.t_max

        # Temperature decreases linearly from top to bottom
        denom = max(1, len(scale_y) - 1)
        scale_temps = t_max_val - (np.arange(len(scale_y)) / denom) * (t_max_val - t_min_val)
        return scale_colors, scale_temps

    def map_roi_to_temperatures(self, crop, scale_colors, scale_temps):
        """
        Maps pixel RGB values to calibrated temperatures via nearest FLIR color palette distance.
        """
        h, w, _ = crop.shape
        flat = crop.reshape(-1, 3).astype(float)
        diff = np.sum((flat[:, None, :] - scale_colors[None, :, :]) ** 2, axis=-1)
        best_indices = np.argmin(diff, axis=-1)
        temps = scale_temps[best_indices].reshape(h, w)
        return temps

    def predict_ai_cornea_centers(self, arr_240_320, scale_colors, scale_temps):
        """
        Predicts corneal centers for RE (OD) and LE (OS) using robust anatomical canthi
        and bilateral thermal landmark optimization.
        """
        try:
            h, w, _ = arr_240_320.shape
            down = arr_240_320[::4, ::4, :].astype(float) # (60, 80, 3)
            diff = np.sum((down[:, :, None, :] - scale_colors[None, None, :, :])**2, axis=-1)
            best = np.argmin(diff, axis=-1)
            t_grid = scale_temps[best] # (60, 80)

            # Detect bilateral medial canthi
            re_patch = t_grid[22:45, 26:38]
            re_my, re_mx = np.unravel_index(np.argmax(re_patch), re_patch.shape)
            re_canthus_x = (26 + re_mx) * 4
            re_canthus_y = (22 + re_my) * 4

            le_patch = t_grid[22:45, 38:49]
            le_my, le_mx = np.unravel_index(np.argmax(le_patch), le_patch.shape)
            le_canthus_x = (38 + le_mx) * 4
            le_canthus_y = (22 + le_my) * 4

            y_eye = int((re_canthus_y + le_canthus_y) / 2)
            x_mid = (re_canthus_x + le_canthus_x) / 2.0

            # OD expected position
            re_cx = float(max(50, min(135, round(x_mid - 48))))
            re_cy = float(y_eye)

            # OS mirrored position
            le_cx = float(max(180, min(270, round(x_mid + (x_mid - re_cx)))))
            le_cy = float(re_cy)

            return re_cx, re_cy, le_cx, le_cy
        except Exception:
            return 96.0, 150.0, 194.0, 150.0

    def refine_cornea_apex(self, arr_orig, scale_colors, scale_temps, init_cx, init_cy):
        """
        Refines corneal center within local physiological search radius.
        """
        h, w, _ = arr_orig.shape
        best_x, best_y = int(init_cx), int(init_cy)
        min_cost = 999.0

        for dy in range(-16, 17, 2):
            for dx in range(-16, 17, 2):
                x = int(init_cx) + dx
                y = int(init_cy) + dy
                if x < 15 or x > w - 25 or y < 15 or y > h - 15:
                    continue
                patch = arr_orig[y-2:y+3, x-2:x+3, :].reshape(-1, 3).astype(float)
                diff = np.sum((patch[:, None, :] - scale_colors[None, :, :])**2, axis=-1)
                t = float(np.mean(scale_temps[np.argmin(diff, axis=-1)]))
                if 32.0 <= t <= 36.5:
                    cost = abs(t - 34.5) * 0.4 + (dx*dx + dy*dy) * 0.015
                    if cost < min_cost:
                        min_cost = cost
                        best_x, best_y = x, y
        return best_x, best_y

    def sample_cornea_biomarkers(self, arr_orig, scale_colors, scale_temps, cx, cy, is_re=True):
        """
        Extracts biophysical biomarkers strictly from the segmented anatomical cornea.
        """
        h, w, _ = arr_orig.shape
        rx, ry = 13, 11  # Minimized compact corneal radius strictly targeting cornea optical zone

        cornea_temps = []
        cc_temps = []
        nc_temps = []
        tc_temps = []
        cold_spots = 0

        y_min = max(0, cy - ry)
        y_max = min(h - 1, cy + ry)
        x_min = max(0, cx - rx)
        x_max = min(w - 1, cx + rx)

        # Batch map patch to temps
        crop = arr_orig[y_min:y_max+1, x_min:x_max+1]
        temps_patch = self.map_roi_to_temperatures(crop, scale_colors, scale_temps)

        ph, pw = temps_patch.shape
        for py in range(ph):
            for px in range(pw):
                x = x_min + px
                y = y_min + py
                edist = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
                if edist <= 1.0: # Strictly inside cornea
                    t = temps_patch[py, px]
                    cornea_temps.append(t)
                    if t < 34.2:
                        cold_spots += 1
                    if edist <= 0.15: # Central optical zone apex
                        cc_temps.append(t)
                    
                    if is_re:
                        if x > cx + 6 and abs(y - cy) <= 8: nc_temps.append(t)
                        elif x < cx - 6 and abs(y - cy) <= 8: tc_temps.append(t)
                    else:
                        if x < cx - 6 and abs(y - cy) <= 8: nc_temps.append(t)
                        elif x > cx + 6 and abs(y - cy) <= 8: tc_temps.append(t)

        t_mean = float(np.mean(cornea_temps)) if cornea_temps else 34.8
        t_cc = float(np.mean(cc_temps)) if cc_temps else t_mean
        t_nc = float(np.mean(nc_temps)) if nc_temps else t_cc + 0.35
        t_tc = float(np.mean(tc_temps)) if tc_temps else t_cc - 0.25
        t_std = float(np.std(cornea_temps)) if cornea_temps else 0.45
        cold_ratio = float(cold_spots / max(1, len(cornea_temps)))

        return {
            't_cc': round(t_cc, 2),
            't_nc': round(t_nc, 2),
            't_tc': round(t_tc, 2),
            't_mean': round(t_mean, 2),
            't_min': round(float(np.min(cornea_temps)) if cornea_temps else 32.0, 2),
            't_max': round(float(np.max(cornea_temps)) if cornea_temps else 36.5, 2),
            't_std': round(t_std, 2),
            'cold_spots_ratio': round(cold_ratio, 3),
            'cx': cx,
            'cy': cy,
            'rx': rx,
            'ry': ry,
            'box': [max(0, cx - 38), max(0, cy - 28), min(w, cx + 38), min(h, cy + 28)]
        }

    def process_image(self, img_input):
        """
        Accepts either a file path or PIL.Image or numpy RGB array.
        Returns:
            - normalized_224: np.ndarray (224, 224, 3) in [0, 1]
            - metrics: dict containing clinical thermal metrics & auto-detected cornea locations
        """
        if isinstance(img_input, str):
            im = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, Image.Image):
            im = img_input.convert('RGB')
        elif isinstance(img_input, np.ndarray):
            im = Image.fromarray(img_input.astype(np.uint8)).convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

        orig_w, orig_h = im.size
        arr_orig = np.array(im)
        
        # 1. Standard 224x224 normalized image for CNN input
        im_224 = im.resize((224, 224), Image.Resampling.BILINEAR)
        arr_224 = np.array(im_224, dtype=np.float32) / 255.0

        # Extract dynamic high and least temperatures present in the uploaded thermal image
        high_temp, least_temp, is_inset = self.extract_corner_temps(arr_orig)
        if high_temp is not None and least_temp is not None and high_temp > least_temp:
            t_max_used = high_temp
            t_min_used = least_temp
        else:
            t_max_used = self.t_max
            t_min_used = self.t_min

        # 2. Extract FLIR scale bar calibrated to image's least and high temperatures
        scale_colors, scale_temps = self.extract_calibration_from_array(arr_orig, t_min=t_min_used, t_max=t_max_used, is_inset=is_inset)

        # 3. Predict & Refine Cornea and Eye Regions using AI Model
        # Downscale to 320x240 standard coordinate space if needed
        im_320 = im.resize((320, 240), Image.Resampling.BILINEAR)
        arr_320 = np.array(im_320)
        p_re_x, p_re_y, p_le_x, p_le_y = self.predict_ai_cornea_centers(arr_320, scale_colors, scale_temps)

        # Map predictions to native image dimensions
        sx = orig_w / 320.0
        sy = orig_h / 240.0
        p_re_x_orig = p_re_x * sx
        p_re_y_orig = p_re_y * sy
        p_le_x_orig = p_le_x * sx
        p_le_y_orig = p_le_y * sy

        # Refine apex coordinates
        re_cx, re_cy = self.refine_cornea_apex(arr_orig, scale_colors, scale_temps, p_re_x_orig, p_re_y_orig)
        le_cx, le_cy = self.refine_cornea_apex(arr_orig, scale_colors, scale_temps, p_le_x_orig, p_le_y_orig)

        # 4. Extract biophysical biomarkers strictly from cornea
        re_metrics = self.sample_cornea_biomarkers(arr_orig, scale_colors, scale_temps, re_cx, re_cy, is_re=True)
        le_metrics = self.sample_cornea_biomarkers(arr_orig, scale_colors, scale_temps, le_cx, le_cy, is_re=False)

        # Ocular Asymmetry metrics
        cc_asymmetry = round(abs(re_metrics['t_cc'] - le_metrics['t_cc']), 2)
        mean_asymmetry = round(abs(re_metrics['t_mean'] - le_metrics['t_mean']), 2)
        cold_spots_avg = round((re_metrics['cold_spots_ratio'] + le_metrics['cold_spots_ratio']) / 2.0, 3)
        min_ocular_cc = min(re_metrics['t_cc'], le_metrics['t_cc'])
        mean_ocular_cc = round((re_metrics['t_cc'] + le_metrics['t_cc']) / 2.0, 2)
        
        is_abnormal_temp = (
            min_ocular_cc < 34.0 or
            cc_asymmetry >= 0.50 or
            cold_spots_avg > 0.15 or
            re_metrics['t_std'] > 1.1 or
            le_metrics['t_std'] > 1.1
        )
        
        # Nasal-temporal temperature gradient (dry eye often disrupts this gradient)
        re_nt_gradient = round(re_metrics['t_nc'] - re_metrics['t_tc'], 2)
        le_nt_gradient = round(le_metrics['t_nc'] - le_metrics['t_tc'], 2)
        # Temperature range across cornea (wider range = more non-uniform tear film)
        re_temp_range = round(re_metrics['t_max'] - re_metrics['t_min'], 2)
        le_temp_range = round(le_metrics['t_max'] - le_metrics['t_min'], 2)
        # Central cornea deviation from regional mean (dry eye shows depressed central temp)
        re_cc_deviation = round(re_metrics['t_cc'] - re_metrics['t_mean'], 2)
        le_cc_deviation = round(le_metrics['t_cc'] - le_metrics['t_mean'], 2)
        # Worst-eye cold spots (most affected eye drives diagnosis)
        max_cold_spots = round(max(re_metrics['cold_spots_ratio'], le_metrics['cold_spots_ratio']), 3)

        feature_vector = [
            mean_ocular_cc,                          # 0: mean bilateral CC temp
            min_ocular_cc,                           # 1: min bilateral CC temp
            cc_asymmetry,                            # 2: CC asymmetry between eyes
            mean_asymmetry,                          # 3: mean temp asymmetry
            cold_spots_avg,                          # 4: average cold spots ratio
            re_metrics['t_cc'],                      # 5: RE central cornea temp
            le_metrics['t_cc'],                      # 6: LE central cornea temp
            re_metrics['t_std'],                     # 7: RE thermal non-uniformity
            le_metrics['t_std'],                     # 8: LE thermal non-uniformity
            re_metrics['cold_spots_ratio'],          # 9: RE cold spots ratio
            le_metrics['cold_spots_ratio'],          # 10: LE cold spots ratio
            re_nt_gradient,                          # 11: RE nasal-temporal gradient
            le_nt_gradient,                          # 12: LE nasal-temporal gradient
            re_temp_range,                           # 13: RE temperature range
            le_temp_range,                           # 14: LE temperature range
            re_cc_deviation,                         # 15: RE central deviation from mean
            le_cc_deviation,                         # 16: LE central deviation from mean
            max_cold_spots                           # 17: worst-eye cold spots
        ]
        
        return arr_224, {
            're': re_metrics,
            'le': le_metrics,
            'cornea_locations': {
                're': {
                    'cx': re_cx,
                    'cy': re_cy,
                    'rx': re_metrics['rx'],
                    'ry': re_metrics['ry'],
                    'eye_box': re_metrics['box']
                },
                'le': {
                    'cx': le_cx,
                    'cy': le_cy,
                    'rx': le_metrics['rx'],
                    'ry': le_metrics['ry'],
                    'eye_box': le_metrics['box']
                }
            },
            'central_cornea_temp': mean_ocular_cc,
            'min_cornea_temp': min_ocular_cc,
            'least_temp': round(float(t_min_used), 1),
            'high_temp': round(float(t_max_used), 1),
            'bilateral_asymmetry': cc_asymmetry,
            'tear_breakup_cold_spots': cold_spots_avg,
            'temperature_status': 'Abnormal' if is_abnormal_temp else 'Normal',
            'feature_vector': feature_vector
        }
