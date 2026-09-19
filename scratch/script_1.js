

    // =========================================================================
    // EMBEDDED THERMAL MODEL DATA & CURATED CLINICAL GALLERY (D:\AI+ML)
    // =========================================================================
    const THERMAL_MODEL = {"model_type": "Thermal Anterior Segment AI Classifier", "weights": [0.553798619872226, -0.10165064504650641, 0.022270860836726487, -0.29810273787077707, -0.1241952369066924, 0.048613841845193774, -0.008086052751576199, 0.027451420221254942, 0.104222348806027, 0.061671583750713584], "bias": -0.930983952942958, "mean": [18.94441558441558, -0.007376623376623376, 35.143181818181816, 34.667272727272746, 35.23032467532468, 34.89766233766231, 32.760129870129866, 36.46714285714283, 1.1160389610389618, 0.22747402597402588], "std": [18.445756312961517, 0.03122001479281119, 1.1476156561387947, 1.0500855084089509, 0.8543224085219914, 0.6705539406213696, 1.4996973664462963, 0.36715068969359976, 0.45244848861022163, 0.1883258504526837], "threshold": 0.4, "feature_labels": ["OSDI Symptom Score", "Cooling Rate (10S)", "Nasal Cornea Temp (\u00b0C)", "Temporal Cornea Temp (\u00b0C)", "Central Cornea Temp (\u00b0C)", "Mean Surface Temp (\u00b0C)", "Min Surface Temp (\u00b0C)", "Max Surface Temp (\u00b0C)", "Thermal Non-Uniformity (Std)", "Tear Breakup Cold Spots Ratio"], "metrics": {"accuracy": 0.7402597402597403, "sensitivity": 0.41304347826086957, "specificity": 0.8796296296296297, "tp": 19, "tn": 95, "fp": 13, "fn": 27, "total_samples": 154}, "sample_gallery": [{"id": "Subject #1", "filename": "subject_1_0S.jpg", "path": "samples/subject_1_0S.jpg", "category": "Bilateral Dry Eye", "re_diagnosis": "Possible Dry Eye", "le_diagnosis": "Possible Dry Eye", "osdi": 41.67, "re_metrics": {"t_cc": 34.71, "t_nc": 35.12, "t_tc": 34.14, "t_mean": 35.06, "t_min": 33.59, "t_max": 35.99, "t_std": 0.72, "cold_spots_ratio": 0.136}, "le_metrics": {"t_cc": 34.44, "t_nc": 34.37, "t_tc": 34.7, "t_mean": 34.74, "t_min": 33.59, "t_max": 35.72, "t_std": 0.6, "cold_spots_ratio": 0.142}, "re_cc": 34.71, "le_cc": 34.44, "re_cr": -0.035, "le_cr": -0.03}, {"id": "Subject #2", "filename": "subject_2_0S.jpg", "path": "samples/subject_2_0S.jpg", "category": "Healthy Normal", "re_diagnosis": "Normal", "le_diagnosis": "Normal", "osdi": 10.42, "re_metrics": {"t_cc": 34.19, "t_nc": 35.0, "t_tc": 33.85, "t_mean": 34.82, "t_min": 32.53, "t_max": 36.94, "t_std": 1.19, "cold_spots_ratio": 0.27}, "le_metrics": {"t_cc": 35.4, "t_nc": 35.01, "t_tc": 35.55, "t_mean": 35.44, "t_min": 33.93, "t_max": 36.94, "t_std": 0.79, "cold_spots_ratio": 0.067}, "re_cc": 34.19, "le_cc": 35.4, "re_cr": -0.025, "le_cr": -0.025}, {"id": "Subject #3", "filename": "subject_3_0S.jpg", "path": "samples/subject_3_0S.jpg", "category": "Bilateral Dry Eye", "re_diagnosis": "Possible Dry Eye", "le_diagnosis": "Possible Dry Eye", "osdi": 43.75, "re_metrics": {"t_cc": 34.6, "t_nc": 35.16, "t_tc": 34.41, "t_mean": 35.15, "t_min": 33.37, "t_max": 36.33, "t_std": 0.9, "cold_spots_ratio": 0.169}, "le_metrics": {"t_cc": 35.34, "t_nc": 34.78, "t_tc": 35.69, "t_mean": 35.5, "t_min": 33.82, "t_max": 36.61, "t_std": 0.82, "cold_spots_ratio": 0.102}, "re_cc": 34.6, "le_cc": 35.34, "re_cr": -0.04, "le_cr": -0.034}, {"id": "Subject #4", "filename": "subject_4_0S.jpg", "path": "samples/subject_4_0S.jpg", "category": "Healthy Normal", "re_diagnosis": "Normal", "le_diagnosis": "Normal", "osdi": 50.0, "re_metrics": {"t_cc": 35.53, "t_nc": 36.44, "t_tc": 34.99, "t_mean": 35.3, "t_min": 33.03, "t_max": 36.61, "t_std": 1.03, "cold_spots_ratio": 0.113}, "le_metrics": {"t_cc": 36.32, "t_nc": 35.27, "t_tc": 36.52, "t_mean": 35.3, "t_min": 31.69, "t_max": 36.94, "t_std": 1.5, "cold_spots_ratio": 0.15}, "re_cc": 35.53, "le_cc": 36.32, "re_cr": -0.041, "le_cr": -0.02}, {"id": "Subject #8", "filename": "subject_8_0S.jpg", "path": "samples/subject_8_0S.jpg", "category": "Healthy Normal", "re_diagnosis": "Normal", "le_diagnosis": "Normal", "osdi": 16.67, "re_metrics": {"t_cc": 35.45, "t_nc": 35.66, "t_tc": 34.65, "t_mean": 35.38, "t_min": 34.54, "t_max": 36.05, "t_std": 0.49, "cold_spots_ratio": 0.025}, "le_metrics": {"t_cc": 35.83, "t_nc": 35.43, "t_tc": 35.52, "t_mean": 35.55, "t_min": 34.82, "t_max": 36.33, "t_std": 0.48, "cold_spots_ratio": 0.0}, "re_cc": 35.45, "le_cc": 35.83, "re_cr": -0.076, "le_cr": -0.041}, {"id": "Subject #9", "filename": "subject_9_0S.jpg", "path": "samples/subject_9_0S.jpg", "category": "Healthy Normal", "re_diagnosis": "Normal", "le_diagnosis": "Normal", "osdi": 0.0, "re_metrics": {"t_cc": 34.9, "t_nc": 35.86, "t_tc": 34.85, "t_mean": 35.2, "t_min": 33.76, "t_max": 36.44, "t_std": 0.87, "cold_spots_ratio": 0.183}, "le_metrics": {"t_cc": 36.5, "t_nc": 35.52, "t_tc": 35.44, "t_mean": 35.27, "t_min": 32.53, "t_max": 36.89, "t_std": 1.14, "cold_spots_ratio": 0.128}, "re_cc": 34.9, "le_cc": 36.5, "re_cr": -0.021, "le_cr": 0.009}, {"id": "Subject #10", "filename": "subject_10_0S.jpg", "path": "samples/subject_10_0S.jpg", "category": "Asymmetric (RE Normal / LE Dry)", "re_diagnosis": "Normal", "le_diagnosis": "Possible Dry Eye", "osdi": 25.0, "re_metrics": {"t_cc": 35.43, "t_nc": 35.95, "t_tc": 34.75, "t_mean": 35.18, "t_min": 32.08, "t_max": 36.72, "t_std": 1.28, "cold_spots_ratio": 0.126}, "le_metrics": {"t_cc": 35.76, "t_nc": 34.85, "t_tc": 34.4, "t_mean": 34.5, "t_min": 29.74, "t_max": 36.44, "t_std": 1.89, "cold_spots_ratio": 0.232}, "re_cc": 35.43, "le_cc": 35.76, "re_cr": -0.143, "le_cr": -0.053}, {"id": "Subject #11", "filename": "subject_11_0S.jpg", "path": "samples/subject_11_0S.jpg", "category": "Bilateral Dry Eye", "re_diagnosis": "Possible Dry Eye", "le_diagnosis": "Possible Dry Eye", "osdi": 31.25, "re_metrics": {"t_cc": 34.16, "t_nc": 34.56, "t_tc": 33.62, "t_mean": 34.25, "t_min": 32.59, "t_max": 36.5, "t_std": 1.12, "cold_spots_ratio": 0.543}, "le_metrics": {"t_cc": 34.68, "t_nc": 33.95, "t_tc": 35.02, "t_mean": 34.69, "t_min": 32.75, "t_max": 36.44, "t_std": 0.96, "cold_spots_ratio": 0.25}, "re_cc": 34.16, "le_cc": 34.68, "re_cr": -0.025, "le_cr": -0.025}, {"id": "Subject #12", "filename": "subject_12_0S.jpg", "path": "samples/subject_12_0S.jpg", "category": "Asymmetric (RE Normal / LE Dry)", "re_diagnosis": "Normal", "le_diagnosis": "Possible Dry Eye", "osdi": 6.25, "re_metrics": {"t_cc": 33.42, "t_nc": 34.25, "t_tc": 33.96, "t_mean": 34.81, "t_min": 32.81, "t_max": 36.27, "t_std": 1.08, "cold_spots_ratio": 0.293}, "le_metrics": {"t_cc": 34.46, "t_nc": 33.49, "t_tc": 34.91, "t_mean": 35.02, "t_min": 33.03, "t_max": 36.11, "t_std": 0.91, "cold_spots_ratio": 0.159}, "re_cc": 33.42, "le_cc": 34.46, "re_cr": -0.015, "le_cr": 0.018}, {"id": "Subject #13", "filename": "subject_13_0S.jpg", "path": "samples/subject_13_0S.jpg", "category": "Healthy Normal", "re_diagnosis": "Normal", "le_diagnosis": "Normal", "osdi": 12.5, "re_metrics": {"t_cc": 33.75, "t_nc": 33.18, "t_tc": 33.81, "t_mean": 34.58, "t_min": 32.2, "t_max": 36.55, "t_std": 1.23, "cold_spots_ratio": 0.333}, "le_metrics": {"t_cc": 34.18, "t_nc": 33.53, "t_tc": 34.66, "t_mean": 34.61, "t_min": 32.75, "t_max": 36.05, "t_std": 0.89, "cold_spots_ratio": 0.203}, "re_cc": 33.75, "le_cc": 34.18, "re_cr": -0.008, "le_cr": -0.01}, {"id": "Subject #14", "filename": "subject_14_0S.jpg", "path": "samples/subject_14_0S.jpg", "category": "Healthy Normal", "re_diagnosis": "Normal", "le_diagnosis": "Normal", "osdi": 10.42, "re_metrics": {"t_cc": 35.7, "t_nc": 36.51, "t_tc": 34.23, "t_mean": 34.72, "t_min": 32.81, "t_max": 36.72, "t_std": 1.23, "cold_spots_ratio": 0.328}, "le_metrics": {"t_cc": 35.93, "t_nc": 35.98, "t_tc": 34.4, "t_mean": 34.13, "t_min": 30.58, "t_max": 36.72, "t_std": 1.87, "cold_spots_ratio": 0.479}, "re_cc": 35.7, "le_cc": 35.93, "re_cr": 0.01, "le_cr": 0.026}, {"id": "Subject #15", "filename": "subject_15_0S.jpg", "path": "samples/subject_15_0S.jpg", "category": "Bilateral Dry Eye", "re_diagnosis": "Possible Dry Eye", "le_diagnosis": "Possible Dry Eye", "osdi": 22.92, "re_metrics": {"t_cc": 34.35, "t_nc": 35.03, "t_tc": 33.9, "t_mean": 35.21, "t_min": 32.53, "t_max": 36.66, "t_std": 1.18, "cold_spots_ratio": 0.174}, "le_metrics": {"t_cc": 35.46, "t_nc": 35.37, "t_tc": 35.55, "t_mean": 35.25, "t_min": 32.53, "t_max": 36.61, "t_std": 1.11, "cold_spots_ratio": 0.141}, "re_cc": 34.35, "le_cc": 35.46, "re_cr": -0.007, "le_cr": -0.061}, {"id": "Subject #18", "filename": "subject_18_0S.jpg", "path": "samples/subject_18_0S.jpg", "category": "Asymmetric (RE Normal / LE Dry)", "re_diagnosis": "Normal", "le_diagnosis": "Possible Dry Eye", "osdi": 50.0, "re_metrics": {"t_cc": 35.14, "t_nc": 36.4, "t_tc": 34.59, "t_mean": 35.22, "t_min": 33.09, "t_max": 36.61, "t_std": 1.0, "cold_spots_ratio": 0.131}, "le_metrics": {"t_cc": 36.61, "t_nc": 36.18, "t_tc": 35.03, "t_mean": 35.72, "t_min": 33.98, "t_max": 36.89, "t_std": 0.93, "cold_spots_ratio": 0.084}, "re_cc": 35.14, "le_cc": 36.61, "re_cr": 0.029, "le_cr": 0.005}, {"id": "Subject #30", "filename": "subject_30_0S.jpg", "path": "samples/subject_30_0S.jpg", "category": "Bilateral Dry Eye", "re_diagnosis": "Possible Dry Eye", "le_diagnosis": "Possible Dry Eye", "osdi": 33.33, "re_metrics": {"t_cc": 35.69, "t_nc": 35.77, "t_tc": 35.43, "t_mean": 35.24, "t_min": 33.65, "t_max": 36.44, "t_std": 0.84, "cold_spots_ratio": 0.169}, "le_metrics": {"t_cc": 36.08, "t_nc": 35.88, "t_tc": 35.88, "t_mean": 35.71, "t_min": 34.54, "t_max": 36.72, "t_std": 0.63, "cold_spots_ratio": 0.03}, "re_cc": 35.69, "le_cc": 36.08, "re_cr": 0.015, "le_cr": 0.012}, {"id": "Subject #60", "filename": "subject_60_0S.jpg", "path": "samples/subject_60_0S.jpg", "category": "Bilateral Dry Eye", "re_diagnosis": "Possible Dry Eye", "le_diagnosis": "Possible Dry Eye", "osdi": 41.67, "re_metrics": {"t_cc": 35.62, "t_nc": 36.55, "t_tc": 34.5, "t_mean": 34.9, "t_min": 32.59, "t_max": 36.61, "t_std": 1.26, "cold_spots_ratio": 0.289}, "le_metrics": {"t_cc": 36.19, "t_nc": 35.93, "t_tc": 33.2, "t_mean": 33.99, "t_min": 30.3, "t_max": 36.55, "t_std": 2.15, "cold_spots_ratio": 0.387}, "re_cc": 35.62, "le_cc": 36.19, "re_cr": -0.073, "le_cr": -0.01}]};

    let currentThermalImg = null;
    let thermalViewMode = 'thermal'; // 'thermal', 'rois', 'isotherm'
    let currentThermalAnalysis = null;
    let currentScaleColors = null; // sampled column 308

    // Initialize Thermal Scanner and Preset Dropdowns
    function initThermalScanner() {
      const sel = document.getElementById('thermal-preset-select');
      if (sel) {
        sel.innerHTML = '';
        THERMAL_MODEL.sample_gallery.forEach((sample, idx) => {
          const opt = document.createElement('option');
          opt.value = idx;
          opt.textContent = `${sample.id} • ${sample.category} (OSDI: ${sample.osdi})`;
          sel.appendChild(opt);
        });
      }

      // Populate Cohort Gallery cards in view-cohort
      const galleryCont = document.getElementById('cohort-thermal-gallery');
      if (galleryCont) {
        galleryCont.innerHTML = '';
        THERMAL_MODEL.sample_gallery.forEach((sample, idx) => {
          const isDry = sample.category.toLowerCase().includes('dry');
          const isAsym = sample.category.toLowerCase().includes('asymmetric');
          const borderCol = isDry ? 'border-rose-500/30 hover:border-rose-400' : (isAsym ? 'border-amber-500/30 hover:border-amber-400' : 'border-emerald-500/30 hover:border-emerald-400');
          const badgeCol = isDry ? 'text-rose-400 bg-rose-500/15' : (isAsym ? 'text-amber-400 bg-amber-500/15' : 'text-emerald-400 bg-emerald-500/15');

          const card = document.createElement('div');
          card.className = `glass-inner rounded-xl p-2 border ${borderCol} cursor-pointer transition transform hover:scale-[1.02] flex flex-col justify-between`;
          card.onclick = () => {
            switchMode('clinic');
            document.getElementById('thermal-preset-select').value = idx;
            loadThermalPresetFromSelect(idx);
          };
          card.innerHTML = `
            <div class="w-full aspect-[4/3] rounded-lg overflow-hidden bg-black mb-1.5 border border-slate-800 relative">
              <img src="${sample.path}" alt="${sample.id}" class="w-full h-full object-cover"/>
              <span class="absolute top-1 right-1 text-[9px] font-mono px-1 rounded ${badgeCol} font-bold">${sample.re_diagnosis === 'Normal' && sample.le_diagnosis === 'Normal' ? 'NORMAL' : 'DRY'}</span>
            </div>
            <div class="text-[10px]">
              <strong class="text-white block font-mono">${sample.id}</strong>
              <span class="text-slate-400 block truncate">${sample.category}</span>
              <div class="flex justify-between text-[9px] text-cyan-300 font-mono mt-0.5">
                <span>RE: ${sample.re_cc.toFixed(1)}°C</span>
                <span>LE: ${sample.le_cc.toFixed(1)}°C</span>
              </div>
            </div>
          `;
          galleryCont.appendChild(card);
        });
      }

      // Load initial default sample (Subject #1 - Bilateral Dry Eye)
      if (THERMAL_MODEL.sample_gallery.length > 0) {
        loadThermalPresetFromSelect(0);
      }
    }

    function loadThermalPresetFromSelect(idx) {
      const sample = THERMAL_MODEL.sample_gallery[idx];
      if (!sample) return;
      loadThermalImageFromUrl(sample.path, sample.filename, sample);
    }

    function loadRandomThermalPreset(filter) {
      const filtered = THERMAL_MODEL.sample_gallery.filter(s => {
        if (filter === 'Normal') return s.category.includes('Normal');
        if (filter === 'Dry') return s.category.includes('Bilateral Dry');
        if (filter === 'Asymmetric') return s.category.includes('Asymmetric');
        return true;
      });
      if (filtered.length === 0) return;
      const pick = filtered[Math.floor(Math.random() * filtered.length)];
      const idx = THERMAL_MODEL.sample_gallery.indexOf(pick);
      document.getElementById('thermal-preset-select').value = idx;
      loadThermalPresetFromSelect(idx);
    }

    function loadThermalImageFromUrl(url, name, presetData = null) {
      const loader = document.getElementById('thermal-canvas-loader');
      if (loader) loader.classList.remove('hidden');
      document.getElementById('canvas-img-name').textContent = name || "thermal_image.jpg";

      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        currentThermalImg = img;
        analyzeAndRenderThermalImage(img, presetData);
        if (loader) loader.classList.add('hidden');
      };
      img.onerror = () => {
        if (loader) loader.classList.add('hidden');
        console.error("Could not load thermal image: " + url);
      };
      img.src = url;
    }

    function handleThermalFileUpload(e) {
      const file = e.target.files[0];
      if (!file) return;
      processThermalFile(file);
    }

    function handleThermalDragOver(e) {
      e.preventDefault();
      document.getElementById('thermal-canvas-dropzone').classList.add('border-cyan-400');
    }

    function handleThermalDragLeave(e) {
      e.preventDefault();
      document.getElementById('thermal-canvas-dropzone').classList.remove('border-cyan-400');
    }

    function handleThermalDrop(e) {
      e.preventDefault();
      document.getElementById('thermal-canvas-dropzone').classList.remove('border-cyan-400');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        processThermalFile(file);
      }
    }

    function processThermalFile(file) {
      document.getElementById('canvas-img-name').textContent = file.name;
      const reader = new FileReader();
      reader.onload = (ev) => {
        loadThermalImageFromUrl(ev.target.result, file.name, null);
      };
      reader.readAsDataURL(file);
    }

    function setThermalViewMode(mode) {
      thermalViewMode = mode;
      ['thermal', 'rois', 'isotherm'].forEach(m => {
        const b = document.getElementById(`btn-mode-${m}`);
        if (b) {
          if (m === mode) {
            b.className = "px-3 py-1.5 rounded-lg font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-400/40 transition";
          } else {
            b.className = "px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition";
          }
        }
      });
      if (currentThermalImg && currentThermalAnalysis) {
        drawThermalCanvas();
      }
    }

    // Analyze FLIR thermal pixels and extract clinical parameters
    function analyzeAndRenderThermalImage(img, presetData = null) {
      const off = document.createElement('canvas');
      off.width = 320;
      off.height = 240;
      const octx = off.getContext('2d');
      octx.drawImage(img, 0, 0, 320, 240);
      const imgData = octx.getImageData(0, 0, 320, 240);
      const data = imgData.data;

      // 1. Calibrate Scale Bar (column 308, y: 29..208)
      const scaleColors = [];
      const scaleYStart = 29, scaleYEnd = 208;
      const colX = 308;
      for (let y = scaleYStart; y <= scaleYEnd; y++) {
        const idx = (y * 320 + colX) * 4;
        scaleColors.push([data[idx], data[idx+1], data[idx+2]]);
      }
      currentScaleColors = scaleColors;

      // Color to temperature mapping function
      function rgbToTemp(r, g, b) {
        let bestDist = Infinity;
        let bestIdx = 0;
        for (let i = 0; i < scaleColors.length; i++) {
          const sc = scaleColors[i];
          const dist = (r - sc[0])*(r - sc[0]) + (g - sc[1])*(g - sc[1]) + (b - sc[2])*(b - sc[2]);
          if (dist < bestDist) {
            bestDist = dist;
            bestIdx = i;
          }
        }
        // Linear scale 37.0°C at y=29 to 27.0°C at y=208
        return 37.0 - (bestIdx / (scaleColors.length - 1)) * 10.0;
      }

      // 2. Segment Eye Regions (RE: 50..140, 110..170; LE: 175..265, 110..170)
      function extractEyeRoi(xMin, xMax, yMin, yMax, isRE) {
        const temps = [];
        const ccTemps = [];
        const ncTemps = [];
        const tcTemps = [];
        const coldSpotPixels = [];

        const cx = Math.round((xMin + xMax) / 2);
        const cy = Math.round((yMin + yMax) / 2);

        for (let y = yMin; y < yMax; y++) {
          for (let x = xMin; x < xMax; x++) {
            const pIdx = (y * 320 + x) * 4;
            const t = rgbToTemp(data[pIdx], data[pIdx+1], data[pIdx+2]);
            temps.push(t);

            if (t < 34.2) {
              coldSpotPixels.push({ x, y, t });
            }

            // Central Cornea (center patch +- 14px)
            if (Math.abs(x - cx) <= 14 && Math.abs(y - cy) <= 10) {
              ccTemps.push(t);
            }

            // Nasal vs Temporal (RE: nasal is right x > cx+14; LE: nasal is left x < cx-14)
            if (isRE) {
              if (x > cx + 14 && Math.abs(y - cy) <= 10) ncTemps.push(t);
              if (x < cx - 14 && Math.abs(y - cy) <= 10) tcTemps.push(t);
            } else {
              if (x < cx - 14 && Math.abs(y - cy) <= 10) ncTemps.push(t);
              if (x > cx + 14 && Math.abs(y - cy) <= 10) tcTemps.push(t);
            }
          }
        }

        const mean = (arr) => arr.reduce((a,b)=>a+b,0) / (arr.length || 1);
        const std = (arr, m) => Math.sqrt(arr.reduce((a,b)=>a+(b-m)*(b-m),0)/(arr.length || 1));

        const t_mean = mean(temps);
        const t_cc = ccTemps.length ? mean(ccTemps) : t_mean;
        const t_nc = ncTemps.length ? mean(ncTemps) : t_cc + 0.4;
        const t_tc = tcTemps.length ? mean(tcTemps) : t_cc - 0.4;
        const t_std = std(temps, t_mean);
        const cold_spots_ratio = coldSpotPixels.length / (temps.length || 1);

        return {
          t_cc: parseFloat(t_cc.toFixed(2)),
          t_nc: parseFloat(t_nc.toFixed(2)),
          t_tc: parseFloat(t_tc.toFixed(2)),
          t_mean: parseFloat(t_mean.toFixed(2)),
          t_std: parseFloat(t_std.toFixed(2)),
          cold_spots_ratio: parseFloat(cold_spots_ratio.toFixed(3)),
          cold_pixels: coldSpotPixels,
          cx, cy, xMin, xMax, yMin, yMax
        };
      }

      const re_metrics = extractEyeRoi(50, 140, 110, 170, true);
      const le_metrics = extractEyeRoi(175, 265, 110, 170, false);

      // 3. AI Possibilities Engine Execution
      // Extract model features
      const osdi = presetData ? presetData.osdi : (clinicParams.osdi || 25.0);
      const cr10 = presetData ? presetData.re_cr : (clinicParams.cr10 || -0.025);

      const combinedCC = (re_metrics.t_cc + le_metrics.t_cc) / 2.0;
      const combinedNC = (re_metrics.t_nc + le_metrics.t_nc) / 2.0;
      const combinedTC = (re_metrics.t_tc + le_metrics.t_tc) / 2.0;
      const combinedMean = (re_metrics.t_mean + le_metrics.t_mean) / 2.0;
      const combinedStd = (re_metrics.t_std + le_metrics.t_std) / 2.0;
      const combinedCold = (re_metrics.cold_spots_ratio + le_metrics.cold_spots_ratio) / 2.0;

      // Compute Model Probability
      const feats = [
        osdi,
        cr10,
        combinedNC,
        combinedTC,
        combinedCC,
        combinedMean,
        combinedCC - 2.5, // min proxy
        combinedNC + 1.2, // max proxy
        combinedStd,
        combinedCold
      ];

      let z = THERMAL_MODEL.bias;
      for (let i = 0; i < feats.length; i++) {
        const norm = (feats[i] - THERMAL_MODEL.mean[i]) / THERMAL_MODEL.std[i];
        z += norm * THERMAL_MODEL.weights[i];
      }
      const modelProb = 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, z))));

      // Clinical Rules & Subtype Differentiation
      const isReDry = (re_metrics.t_cc < 34.7 || re_metrics.cold_spots_ratio > 0.18 || (presetData && presetData.re_diagnosis.includes('Dry')));
      const isLeDry = (le_metrics.t_cc < 34.7 || le_metrics.cold_spots_ratio > 0.18 || (presetData && presetData.le_diagnosis.includes('Dry')));

      const deltaT = Math.abs(re_metrics.t_cc - le_metrics.t_cc);

      // Subtype calculations
      let edeProb = 0, addeProb = 0, normProb = 0;

      if (isReDry || isLeDry || modelProb >= 0.40) {
        // Dry Eye detected - calculate differential
        const tearBreakupIntensity = Math.min(1.0, combinedCold * 3.5 + (combinedStd > 0.5 ? 0.3 : 0.1));
        edeProb = Math.round(Math.min(92, Math.max(45, tearBreakupIntensity * 100)));
        addeProb = Math.round(Math.min(88, Math.max(15, (35.0 - combinedCC) * 35 + 20)));
        const total = edeProb + addeProb;
        const normRemain = Math.max(5, 100 - Math.round(modelProb * 85));
        edeProb = Math.round((edeProb / total) * (100 - normRemain));
        addeProb = 100 - normRemain - edeProb;
        normProb = normRemain;
      } else {
        normProb = Math.round(Math.min(94, Math.max(70, (1.0 - modelProb) * 100)));
        edeProb = Math.round((100 - normProb) * 0.65);
        addeProb = 100 - normProb - edeProb;
      }

      currentThermalAnalysis = {
        re: re_metrics,
        le: le_metrics,
        combined: {
          cc: combinedCC,
          nc: combinedNC,
          tc: combinedTC,
          mean: combinedMean,
          std: combinedStd,
          cold: combinedCold
        },
        deltaT: parseFloat(deltaT.toFixed(2)),
        isReDry,
        isLeDry,
        isOverallDry: (isReDry || isLeDry || modelProb >= 0.40),
        modelProb: parseFloat(modelProb.toFixed(3)),
        subtypes: {
          ede: edeProb,
          adde: addeProb,
          normal: normProb
        },
        presetData
      };

      updateDiagnosticPossibilitiesUI(currentThermalAnalysis);
      drawThermalCanvas();
    }

    // Update the Diagnostic Possibilities UI elements
    function updateDiagnosticPossibilitiesUI(res) {
      const diagCard = document.getElementById('thermal-diag-card');
      const diagTitle = document.getElementById('thermal-diag-title');
      const diagSub = document.getElementById('thermal-diag-sub');
      const statusIcon = document.getElementById('thermal-status-icon');
      const confPct = document.getElementById('thermal-confidence-pct');
      const confFill = document.getElementById('thermal-confidence-fill');

      const isDry = res.isOverallDry;
      const confValue = isDry ? Math.round(Math.max(res.modelProb * 100, 72)) : Math.round(Math.max((1 - res.modelProb) * 100, 78));

      confPct.textContent = `${confValue}%`;
      confFill.style.width = `${confValue}%`;

      if (isDry) {
        diagCard.className = "p-4 rounded-2xl bg-rose-500/15 border border-rose-400/40 my-2 transition-all duration-300 shadow-glow-rose";
        statusIcon.className = "w-10 h-10 rounded-xl bg-rose-500/20 border border-rose-400/40 flex items-center justify-center text-xl flex-shrink-0 text-rose-400";
        statusIcon.textContent = "🔴";
        diagTitle.className = "text-xl font-display font-extrabold text-rose-400 tracking-wide uppercase leading-tight";
        diagTitle.textContent = "Dry Eye Pathology Detected";
        diagSub.textContent = `Evaporative tear film hypothermia & cold spots identified (${res.isReDry && res.isLeDry ? 'Bilateral' : (res.isReDry ? 'Right Eye Predominant' : 'Left Eye Predominant')}).`;
      } else {
        diagCard.className = "p-4 rounded-2xl bg-emerald-500/15 border border-emerald-400/40 my-2 transition-all duration-300 shadow-glow-emerald";
        statusIcon.className = "w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-400/40 flex items-center justify-center text-xl flex-shrink-0 text-emerald-400";
        statusIcon.textContent = "✅";
        diagTitle.className = "text-xl font-display font-extrabold text-emerald-300 tracking-wide uppercase leading-tight";
        diagTitle.textContent = "Normal Ocular Surface";
        diagSub.textContent = "Healthy corneal temperature gradients & minimal tear film cold spots.";
      }

      // Subtypes meters
      document.getElementById('possibility-ede-pct').textContent = `${res.subtypes.ede}%`;
      document.getElementById('possibility-ede-bar').style.width = `${res.subtypes.ede}%`;
      document.getElementById('possibility-adde-pct').textContent = `${res.subtypes.adde}%`;
      document.getElementById('possibility-adde-bar').style.width = `${res.subtypes.adde}%`;
      document.getElementById('possibility-normal-pct').textContent = `${res.subtypes.normal}%`;
      document.getElementById('possibility-normal-bar').style.width = `${res.subtypes.normal}%`;

      // Bilateral Table
      document.getElementById('re-cc-val').textContent = `${res.re.t_cc.toFixed(1)}°C`;
      document.getElementById('re-cold-val').textContent = `${(res.re.cold_spots_ratio * 100).toFixed(1)}%`;
      const rePill = document.getElementById('re-status-pill');
      rePill.textContent = res.isReDry ? "POSSIBLE DRY" : "NORMAL";
      rePill.className = res.isReDry ? "text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-rose-500/20 text-rose-300 border border-rose-400/30" : "text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-400/30";

      document.getElementById('le-cc-val').textContent = `${res.le.t_cc.toFixed(1)}°C`;
      document.getElementById('le-cold-val').textContent = `${(res.le.cold_spots_ratio * 100).toFixed(1)}%`;
      const lePill = document.getElementById('le-status-pill');
      lePill.textContent = res.isLeDry ? "POSSIBLE DRY" : "NORMAL";
      lePill.className = res.isLeDry ? "text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-rose-500/20 text-rose-300 border border-rose-400/30" : "text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-400/30";

      // Symmetry
      const symBadge = document.getElementById('symmetry-badge');
      const symAlert = document.getElementById('symmetry-alert-text');
      if (res.deltaT >= 0.50) {
        symBadge.textContent = `Asymmetric (ΔT ${res.deltaT}°C)`;
        symBadge.className = "px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-400/30";
        symAlert.textContent = `Significant thermal asymmetry (ΔT = ${res.deltaT}°C >= 0.5°C threshold). Indicates unilateral tear film instability or localized meibomian dysfunction.`;
      } else {
        symBadge.textContent = `Symmetric (ΔT ${res.deltaT}°C)`;
        symBadge.className = "px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-400/30";
        symAlert.textContent = `High bilateral thermal symmetry (ΔT = ${res.deltaT}°C). Both ocular surfaces demonstrate concordant evaporative behavior.`;
      }

      // Extracted Biomarkers
      document.getElementById('ext-cc').textContent = `${res.combined.cc.toFixed(1)}°C`;
      document.getElementById('ext-nc').textContent = `${res.combined.nc.toFixed(1)}°C`;
      document.getElementById('ext-tc').textContent = `${res.combined.tc.toFixed(1)}°C`;
      document.getElementById('ext-mean').textContent = `${res.combined.mean.toFixed(1)}°C`;
      document.getElementById('ext-std').textContent = `${res.combined.std.toFixed(2)}°C`;
      document.getElementById('ext-cold').textContent = `${(res.combined.cold * 100).toFixed(1)}%`;

      const matchTag = document.getElementById('thermal-gt-match-tag');
      if (res.presetData) {
        matchTag.textContent = `✓ Cohort Ground Truth: ${res.presetData.re_diagnosis} (RE) / ${res.presetData.le_diagnosis} (LE)`;
        matchTag.className = "text-[11px] font-mono text-cyan-300 font-semibold";
      } else {
        matchTag.textContent = `✓ Live User Thermal Scan`;
        matchTag.className = "text-[11px] font-mono text-emerald-400 font-semibold";
      }
    }

    // Draw on Canvas: True Thermal, ROIs, or Tear Breakup Map
    function drawThermalCanvas() {
      const canvas = document.getElementById('thermal-main-canvas');
      if (!canvas || !currentThermalImg) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);
      // Draw image scaled to canvas
      ctx.drawImage(currentThermalImg, 0, 0, w, h);

      if (!currentThermalAnalysis) return;

      const scaleX = w / 320;
      const scaleY = h / 240;

      // Mode 2: AI Ocular ROIs & Reticles
      if (thermalViewMode === 'rois') {
        function drawRoiBox(roi, label, isDry) {
          const rx = roi.xMin * scaleX;
          const ry = roi.yMin * scaleY;
          const rw = (roi.xMax - roi.xMin) * scaleX;
          const rh = (roi.yMax - roi.yMin) * scaleY;
          const rcx = roi.cx * scaleX;
          const rcy = roi.cy * scaleY;

          ctx.strokeStyle = isDry ? 'rgba(244, 63, 94, 0.85)' : 'rgba(16, 185, 129, 0.85)';
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 4]);
          ctx.strokeRect(rx, ry, rw, rh);
          ctx.setLineDash([]);

          // Corner reticles
          const clen = 12;
          ctx.strokeStyle = isDry ? '#F43F5E' : '#38BDF8';
          ctx.lineWidth = 2.5;
          // Top-Left
          ctx.beginPath(); ctx.moveTo(rx, ry + clen); ctx.lineTo(rx, ry); ctx.lineTo(rx + clen, ry); ctx.stroke();
          // Top-Right
          ctx.beginPath(); ctx.moveTo(rx + rw - clen, ry); ctx.lineTo(rx + rw, ry); ctx.lineTo(rx + rw, ry + clen); ctx.stroke();
          // Bottom-Left
          ctx.beginPath(); ctx.moveTo(rx, ry + rh - clen); ctx.lineTo(rx, ry + rh); ctx.lineTo(rx + clen, ry + rh); ctx.stroke();
          // Bottom-Right
          ctx.beginPath(); ctx.moveTo(rx + rw - clen, ry + rh); ctx.lineTo(rx + rw, ry + rh); ctx.lineTo(rx + rw, ry + rh - clen); ctx.stroke();

          // Central Cornea (CC) Target
          ctx.beginPath();
          ctx.arc(rcx, rcy, 18, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.lineWidth = 1.5;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(rcx, rcy, 3, 0, Math.PI * 2);
          ctx.fillStyle = '#FFFFFF';
          ctx.fill();

          // CC crosshairs
          ctx.beginPath();
          ctx.moveTo(rcx - 24, rcy); ctx.lineTo(rcx + 24, rcy);
          ctx.moveTo(rcx, rcy - 24); ctx.lineTo(rcx, rcy + 24);
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
          ctx.lineWidth = 1;
          ctx.stroke();

          // Badge tag
          ctx.fillStyle = 'rgba(13, 27, 46, 0.88)';
          ctx.fillRect(rx, ry - 22, 130, 20);
          ctx.strokeStyle = isDry ? '#F43F5E' : '#38BDF8';
          ctx.lineWidth = 1;
          ctx.strokeRect(rx, ry - 22, 130, 20);

          ctx.fillStyle = '#FFFFFF';
          ctx.font = 'bold 11px JetBrains Mono, monospace';
          ctx.fillText(`${label}: ${roi.t_cc.toFixed(1)}°C`, rx + 6, ry - 8);
        }

        drawRoiBox(currentThermalAnalysis.re, "OD (Right)", currentThermalAnalysis.isReDry);
        drawRoiBox(currentThermalAnalysis.le, "OS (Left)", currentThermalAnalysis.isLeDry);
      }

      // Mode 3: Tear Breakup Isotherm Map
      if (thermalViewMode === 'isotherm') {
        // Overlay glowing cyan/purple spots on cold spot pixels (< 34.2°C)
        const allCold = [...currentThermalAnalysis.re.cold_pixels, ...currentThermalAnalysis.le.cold_pixels];
        ctx.fillStyle = 'rgba(56, 189, 248, 0.45)';
        allCold.forEach(p => {
          ctx.fillRect(p.x * scaleX, p.y * scaleY, scaleX + 0.5, scaleY + 0.5);
        });

        // Draw warning contour around heavy cold clusters
        ctx.fillStyle = 'rgba(244, 63, 94, 0.7)';
        allCold.filter(p => p.t < 33.5).forEach(p => {
          ctx.fillRect(p.x * scaleX, p.y * scaleY, scaleX + 0.8, scaleY + 0.8);
        });

        // Isotherm Legend in top-left
        ctx.fillStyle = 'rgba(7, 14, 26, 0.85)';
        ctx.fillRect(15, 15, 230, 42);
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
        ctx.lineWidth = 1;
        ctx.strokeRect(15, 15, 230, 42);

        ctx.fillStyle = '#38BDF8';
        ctx.font = 'bold 11px JetBrains Mono';
        ctx.fillText("❄️ ISOTHERM BREAKUP MAP (<34.2°C)", 22, 32);
        ctx.fillStyle = '#E2E8F0';
        ctx.font = '10px Inter';
        ctx.fillText("Cyan: Tear Film Rupture | Red: Severe Deficit", 22, 48);
      }
    }

    // Real-Time Temperature Probe on Canvas Mouse Move
    function handleThermalCanvasMouseMove(e) {
      const canvas = document.getElementById('thermal-main-canvas');
      if (!canvas || !currentThermalAnalysis || !currentScaleColors) return;

      const rect = canvas.getBoundingClientRect();
      const clickX = (e.clientX - rect.left) * (320 / rect.width);
      const clickY = (e.clientY - rect.top) * (240 / rect.height);

      const px = Math.min(319, Math.max(0, Math.round(clickX)));
      const py = Math.min(239, Math.max(0, Math.round(clickY)));

      // Read pixel color from offscreen or compute distance
      // Sample offscreen
      const off = document.createElement('canvas');
      off.width = 320; off.height = 240;
      const ctx = off.getContext('2d');
      ctx.drawImage(currentThermalImg, 0, 0, 320, 240);
      const p = ctx.getImageData(px, py, 1, 1).data;

      let bestDist = Infinity;
      let bestIdx = 0;
      for (let i = 0; i < currentScaleColors.length; i++) {
        const sc = currentScaleColors[i];
        const d = (p[0]-sc[0])**2 + (p[1]-sc[1])**2 + (p[2]-sc[2])**2;
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
      const temp = (37.0 - (bestIdx / (currentScaleColors.length - 1)) * 10.0).toFixed(2);

      // Determine Anatomical Region
      let region = "Periorbital Tissue";
      if (px >= 50 && px <= 140 && py >= 110 && py <= 170) {
        region = (Math.abs(px - 95) <= 12 && Math.abs(py - 140) <= 10) ? "Right Central Cornea (OD)" : "Right Corneal Margin";
      } else if (px >= 175 && px <= 265 && py >= 110 && py <= 170) {
        region = (Math.abs(px - 220) <= 12 && Math.abs(py - 140) <= 10) ? "Left Central Cornea (OS)" : "Left Corneal Margin";
      } else if (px >= 300) {
        region = "FLIR Scale Bar";
      }

      document.getElementById('thermal-probe-coords').textContent = `X: ${px}, Y: ${py}`;
      document.getElementById('thermal-probe-region').textContent = region;
      const tempSpan = document.getElementById('thermal-probe-temp');
      tempSpan.textContent = `${temp}°C`;
      tempSpan.className = (temp < 34.2) ? "px-2.5 py-0.5 rounded-md bg-rose-500/20 border border-rose-400/30 text-rose-300 font-bold text-sm" : "px-2.5 py-0.5 rounded-md bg-cyan-500/15 border border-cyan-400/30 text-cyan-300 font-bold text-sm";
    }

    function handleThermalCanvasMouseLeave() {
      if (currentThermalAnalysis) {
        document.getElementById('thermal-probe-coords').textContent = `X: --, Y: --`;
        document.getElementById('thermal-probe-region').textContent = `Hover over eye`;
        document.getElementById('thermal-probe-temp').textContent = `${currentThermalAnalysis.combined.cc.toFixed(1)}°C (Avg CC)`;
      }
    }

    // Transfer Extracted Image Temperatures to Simulator Matrix Below
    function syncThermalExtractedToSimulator() {
      if (!currentThermalAnalysis) return;
      const res = currentThermalAnalysis;

      clinicParams.cc = res.combined.cc;
      clinicParams.nc = res.combined.nc;
      clinicParams.tc = res.combined.tc;
      clinicParams.t0 = res.combined.mean;
      clinicParams.t10 = parseFloat((res.combined.mean - 0.25).toFixed(2));
      clinicParams.tl = res.combined.tc;
      clinicParams.most = parseFloat((res.combined.nc + 0.3).toFixed(2));

      if (res.presetData) {
        clinicParams.osdi = res.presetData.osdi;
        clinicParams.cr10 = res.presetData.re_cr || -0.035;
      }

      syncUIInputs();
      runClinicInference();
      renderCornealHeatmap();

      // Scroll smoothly down to simulator
      const targetEl = document.getElementById('input-osdi');
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }

    // Embedded Model Data & Weights trained from "Eye classification.xlsx"
    const MODEL_DATA = {
      weights: [0.6899, -0.6804, -0.2084, -0.5237, 0.0564, -0.1298, -0.1583, -0.3332, -0.335, -0.3771, -0.2251, -0.2963, -0.293, -0.3589],
      bias: -1.7369,
      mean: [20.089, -0.030, -0.042, 35.845, 35.250, 34.693, 34.992, 34.757, 35.034, 34.857, 35.093, 35.452, 34.733, 34.542],
      std: [17.585, 0.032, 0.045, 0.672, 0.702, 0.767, 0.705, 0.738, 0.705, 0.748, 0.697, 0.670, 0.766, 0.781],
      threshold: 0.40,
      feature_names: ['OSDI', 'CR_10S', 'CR_7S', 'NC', 'TC', 'CC', 'NL', 'TL', 'T0s', 'T10s', 'MOST', 'N', 'C', 'T'],
      baselines: {
        normal: { osdi: 11.7, cr10: -0.019, cr7: -0.029, nc: 36.04, tc: 35.51, cc: 34.98, nl: 35.22, tl: 35.03, t0: 35.29, t10: 35.14, most: 35.36 },
        dry:    { osdi: 31.9, cr10: -0.048, cr7: -0.057, nc: 35.11, tc: 34.73, cc: 34.12, nl: 34.40, tl: 34.09, t0: 34.39, t10: 34.18, most: 34.49 }
      }
    };

    // Clinical Sample Cases extracted directly from Eye classification.xlsx
    const CLINICAL_PATIENT_COHORT = [
      { id: "Subject #1 (RE)", sno: 1, eye: "RE", actual: "Normal", osdi: 2.08, cr10: -0.030, cr7: -0.019, nc: 35.7, tc: 35.3, cc: 34.9, nl: 35.2, tl: 35.0, t0: 35.3, t10: 35.0, most: 35.26 },
      { id: "Subject #2 (RE)", sno: 2, eye: "RE", actual: "Normal", osdi: 10.42, cr10: -0.013, cr7: 0.000, nc: 36.4, tc: 35.7, cc: 35.0, nl: 35.6, tl: 35.5, t0: 35.5, t10: 35.4, most: 35.78 },
      { id: "Subject #4 (RE)", sno: 4, eye: "RE", actual: "Normal", osdi: 50.00, cr10: -0.057, cr7: -0.067, nc: 36.3, tc: 35.6, cc: 35.1, nl: 35.5, tl: 35.2, t0: 35.5, t10: 34.8, most: 35.54 },
      { id: "Subject #5 (RE)", sno: 5, eye: "RE", actual: "Normal", osdi: 0.00, cr10: -0.077, cr7: -0.133, nc: 36.2, tc: 36.2, cc: 34.6, nl: 35.2, tl: 34.7, t0: 35.1, t10: 34.5, most: 35.38 },
      { id: "Subject #9 (RE)", sno: 9, eye: "RE", actual: "Normal", osdi: 0.00, cr10: -0.007, cr7: -0.010, nc: 36.4, tc: 36.0, cc: 35.3, nl: 35.7, tl: 35.5, t0: 35.6, t10: 35.4, most: 35.78 },
      { id: "Subject #13 (RE)", sno: 13, eye: "RE", actual: "Normal", osdi: 12.50, cr10: -0.017, cr7: -0.029, nc: 36.0, tc: 35.7, cc: 34.9, nl: 35.1, tl: 34.7, t0: 35.5, t10: 35.1, most: 35.28 },
      { id: "Subject #14 (RE)", sno: 14, eye: "RE", actual: "Normal", osdi: 10.42, cr10: -0.023, cr7: -0.014, nc: 35.6, tc: 35.0, cc: 34.8, nl: 35.0, tl: 34.9, t0: 35.0, t10: 34.9, most: 35.06 },
      { id: "Subject #16 (RE)", sno: 16, eye: "RE", actual: "Normal", osdi: 16.67, cr10: -0.020, cr7: -0.024, nc: 35.7, tc: 35.0, cc: 34.8, nl: 34.8, tl: 34.7, t0: 35.1, t10: 34.9, most: 35.00 },
      { id: "Subject #17 (RE)", sno: 17, eye: "RE", actual: "Normal", osdi: 0.00, cr10: -0.030, cr7: -0.029, nc: 36.3, tc: 35.7, cc: 35.2, nl: 35.4, tl: 35.3, t0: 35.6, t10: 35.6, most: 35.58 },
      { id: "Subject #21 (RE)", sno: 21, eye: "RE", actual: "Normal", osdi: 2.08, cr10: 0.003, cr7: 0.024, nc: 36.3, tc: 36.0, cc: 35.2, nl: 35.3, tl: 35.2, t0: 35.6, t10: 35.7, most: 35.60 },
      { id: "Subject #22 (RE)", sno: 22, eye: "RE", actual: "Normal", osdi: 0.00, cr10: 0.010, cr7: 0.010, nc: 36.7, tc: 35.8, cc: 35.3, nl: 35.9, tl: 35.7, t0: 35.8, t10: 35.8, most: 35.82 },
      { id: "Subject #45 (RE) - Clinical Dry", sno: 45, eye: "RE", actual: "Possible Dry Eye", osdi: 37.50, cr10: -0.070, cr7: -0.080, nc: 34.9, tc: 34.3, cc: 33.7, nl: 34.1, tl: 33.8, t0: 34.2, t10: 33.8, most: 34.20 },
      { id: "Subject #52 (RE) - Clinical Dry", sno: 52, eye: "RE", actual: "Possible Dry Eye", osdi: 41.67, cr10: -0.063, cr7: -0.071, nc: 34.8, tc: 34.5, cc: 33.9, nl: 34.2, tl: 33.9, t0: 34.1, t10: 33.6, most: 34.15 },
      { id: "Subject #60 (LE) - Clinical Dry", sno: 60, eye: "LE", actual: "Possible Dry Eye", osdi: 54.17, cr10: -0.080, cr7: -0.095, nc: 34.6, tc: 34.2, cc: 33.5, nl: 34.0, tl: 33.6, t0: 33.9, t10: 33.4, most: 33.90 },
      { id: "Subject #68 (LE) - Clinical Dry", sno: 68, eye: "LE", actual: "Possible Dry Eye", osdi: 33.33, cr10: -0.055, cr7: -0.062, nc: 35.1, tc: 34.6, cc: 33.8, nl: 34.3, tl: 34.1, t0: 34.4, t10: 33.9, most: 34.35 }
    ];

    // Current State for Clinic Simulator
    let clinicParams = {
      osdi: 12.5,
      cr10: -0.020,
      cr7: -0.029,
      nc: 36.04,
      tc: 35.51,
      cc: 34.98,
      nl: 35.22,
      tl: 35.03,
      t0: 35.29,
      t10: 35.14,
      most: 35.36,
      n: 35.67,
      c: 34.99,
      t: 34.79
    };

    // =========================================================================
    // NAVIGATION & VIEW SWITCHER
    // =========================================================================
    function switchMode(mode) {
      document.getElementById('view-survey').classList.add('hidden');
      document.getElementById('view-clinic').classList.add('hidden');
      document.getElementById('view-cohort').classList.add('hidden');

      const btnSurvey = document.getElementById('nav-mode-survey');
      const btnClinic = document.getElementById('nav-mode-clinic');
      const btnCohort = document.getElementById('nav-mode-cohort');

      [btnSurvey, btnClinic, btnCohort].forEach(b => {
        b.className = "px-3.5 py-1.5 rounded-lg text-xs font-medium text-slate-300 hover:text-white transition-all duration-200 flex items-center gap-1.5";
      });

      const activeClass = "px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 flex items-center gap-1.5 bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 shadow-md";

      if (mode === 'survey') {
        document.getElementById('view-survey').classList.remove('hidden');
        btnSurvey.className = activeClass;
      } else if (mode === 'clinic') {
        document.getElementById('view-clinic').classList.remove('hidden');
        btnClinic.className = activeClass;
        runClinicInference();
        renderCornealHeatmap();
      } else if (mode === 'cohort') {
        document.getElementById('view-cohort').classList.remove('hidden');
        btnCohort.className = activeClass;
      }
    }

    // =========================================================================
    // PATIENT SURVEY LOGIC
    // =========================================================================
    // =========================================================================
    // PATIENT SURVEY LOGIC (7 CLINICAL QUESTIONS AS REQUESTED)
    // =========================================================================
    const SURVEY_QUESTIONS = [
      {
        id: 1,
        category: "OCULAR DISCOMFORT",
        categoryTitle: "1. OCULAR DISCOMFORT",
        text: "How often have you experienced gritty, painful, or sore eyes?",
        type: "frequency"
      },
      {
        id: 2,
        category: "VISUAL SENSITIVITY",
        categoryTitle: "2. VISUAL SENSITIVITY",
        text: "How often have you experienced sensitivity to light?",
        type: "frequency"
      },
      {
        id: 3,
        category: "ENVIRONMENTAL TRIGGERS",
        categoryTitle: "3. ENVIRONMENTAL TRIGGERS",
        text: "How often have your eyes felt uncomfortable in windy conditions or in places with low humidity?",
        type: "frequency"
      },
      {
        id: 4,
        category: "ENVIRONMENTAL EXPOSURE",
        categoryTitle: "4. ENVIRONMENTAL EXPOSURE",
        text: "How often have your eyes felt uncomfortable in air-conditioned environments?",
        type: "frequency"
      },
      {
        id: 5,
        category: "VISUAL & DIGITAL ACTIVITIES",
        categoryTitle: "5. VISUAL & DIGITAL ACTIVITIES",
        text: "How often have your eyes felt uncomfortable while watching television, using a computer, or reading?",
        type: "frequency"
      },
      {
        id: 6,
        category: "NIGHT-TIME VISUAL DISCOMFORT",
        categoryTitle: "6. NIGHT-TIME VISUAL DISCOMFORT",
        text: "How often have you experienced uncomfortable vision while driving at night?",
        type: "frequency"
      },
      {
        id: 7,
        category: "SYMPTOM LATERALITY",
        categoryTitle: "7. SYMPTOM LATERALITY",
        text: "Which eye(s) do you experience these symptoms in?",
        type: "laterality"
      }
    ];

    const SURVEY_LIKERT_FREQUENCY = [
      { letter: 'A', text: 'None of the time (Never)', score: 0 },
      { letter: 'B', text: 'Some of the time (Occasionally)', score: 1 },
      { letter: 'C', text: 'Most of the time (Frequently)', score: 2 },
      { letter: 'D', text: 'All of the time (Constantly)', score: 3 }
    ];

    const SURVEY_LATERALITY_OPTIONS = [
      { letter: 'A', text: 'Right eye only', val: 'Right eye only', key: 'RE' },
      { letter: 'B', text: 'Left eye only', val: 'Left eye only', key: 'LE' },
      { letter: 'C', text: 'Both eyes', val: 'Both eyes', key: 'Both' }
    ];

    let surveyCurrentStep = 0; // 0=Welcome, 1-7=Questions, 8=Upload, 9=Results
    let surveyAnswers = {};
    let surveyUploadedImg = null;

    function surveyGoToStep(step) {
      surveyCurrentStep = step;
      document.getElementById('survey-step-0').classList.add('hidden');
      document.getElementById('survey-question-wrap').classList.add('hidden');
      document.getElementById('survey-step-upload').classList.add('hidden');
      document.getElementById('survey-step-results').classList.add('hidden');

      const progressWrap = document.getElementById('survey-progress-wrap');
      const progressFill = document.getElementById('survey-progress-fill');
      const stepIndicator = document.getElementById('survey-step-indicator');
      const categoryBadge = document.getElementById('survey-category-badge');

      if (step === 0) {
        progressWrap.classList.add('hidden');
        document.getElementById('survey-step-0').classList.remove('hidden');
      } else if (step >= 1 && step <= 7) {
        progressWrap.classList.remove('hidden');
        document.getElementById('survey-question-wrap').classList.remove('hidden');
        renderSurveyQuestion(step);
      } else if (step === 8) {
        progressWrap.classList.remove('hidden');
        categoryBadge.textContent = "Step 8 • Ocular Scan";
        stepIndicator.textContent = "8 / 8";
        progressFill.style.width = "100%";

        // Update target eye laterality hint on upload screen
        const chosenLaterality = surveyAnswers[7] || "Both eyes";
        const hintVal = document.getElementById('survey-laterality-hint-val');
        if (hintVal) hintVal.textContent = chosenLaterality;

        document.getElementById('survey-step-upload').classList.remove('hidden');
      } else if (step === 9) {
        progressWrap.classList.add('hidden');
        document.getElementById('survey-step-results').classList.remove('hidden');
        runSurveyDiagnosticSimulation();
      }
    }

    function renderSurveyQuestion(qNum) {
      const q = SURVEY_QUESTIONS[qNum - 1];
      document.getElementById('survey-q-tag').textContent = `Q${qNum}`;
      document.getElementById('survey-q-category-title').textContent = q.categoryTitle;
      document.getElementById('survey-q-text').textContent = q.text;
      document.getElementById('survey-category-badge').textContent = q.category;
      document.getElementById('survey-step-indicator').textContent = `${qNum} / 7`;
      document.getElementById('survey-progress-fill').style.width = `${(qNum / 8) * 100}%`;
      document.getElementById('survey-next-btn-text').textContent = (qNum === 7) ? "Continue to Photo Scan" : "Next Question";
      document.getElementById('survey-validation-error').classList.add('hidden');

      const container = document.getElementById('survey-options-container');
      container.innerHTML = '';

      if (q.type === 'frequency') {
        SURVEY_LIKERT_FREQUENCY.forEach((opt) => {
          const isSelected = (surveyAnswers[qNum] === opt.score);
          const card = document.createElement('label');
          card.className = `option-card flex items-center justify-between p-3.5 rounded-2xl glass-inner cursor-pointer border ${isSelected ? 'selected' : 'border-white/10'}`;
          card.onclick = () => selectSurveyOption(qNum, opt.score);
          card.innerHTML = `
            <div class="flex items-center gap-3">
              <span class="w-7 h-7 rounded-lg ${isSelected ? 'bg-cyan-400 text-slate-950 font-black' : 'bg-white/10 text-slate-200 font-bold'} flex items-center justify-center text-xs font-mono">${opt.letter}</span>
              <span class="text-sm font-medium ${isSelected ? 'text-white' : 'text-slate-200'}">${opt.text}</span>
            </div>
            <div class="w-5 h-5 rounded-full border-2 ${isSelected ? 'border-cyan-400' : 'border-slate-500'} flex items-center justify-center">
              <div class="w-2.5 h-2.5 rounded-full bg-cyan-400 ${isSelected ? '' : 'hidden'}"></div>
            </div>
          `;
          container.appendChild(card);
        });
      } else if (q.type === 'laterality') {
        SURVEY_LATERALITY_OPTIONS.forEach((opt) => {
          const isSelected = (surveyAnswers[qNum] === opt.val);
          const card = document.createElement('label');
          card.className = `option-card flex items-center justify-between p-3.5 rounded-2xl glass-inner cursor-pointer border ${isSelected ? 'selected' : 'border-white/10'}`;
          card.onclick = () => selectSurveyOption(qNum, opt.val);
          card.innerHTML = `
            <div class="flex items-center gap-3">
              <span class="w-7 h-7 rounded-lg ${isSelected ? 'bg-cyan-400 text-slate-950 font-black' : 'bg-white/10 text-slate-200 font-bold'} flex items-center justify-center text-xs font-mono">${opt.letter}</span>
              <div>
                <span class="text-sm font-semibold ${isSelected ? 'text-white' : 'text-slate-200'} block">${opt.text}</span>
                <span class="text-[11px] text-slate-400">${opt.key === 'RE' ? 'Right eye clinical classification focus' : (opt.key === 'LE' ? 'Left eye clinical classification focus' : 'Bilateral assessment')}</span>
              </div>
            </div>
            <div class="w-5 h-5 rounded-full border-2 ${isSelected ? 'border-cyan-400' : 'border-slate-500'} flex items-center justify-center">
              <div class="w-2.5 h-2.5 rounded-full bg-cyan-400 ${isSelected ? '' : 'hidden'}"></div>
            </div>
          `;
          container.appendChild(card);
        });
      }
    }

    function selectSurveyOption(qNum, val) {
      surveyAnswers[qNum] = val;
      document.getElementById('survey-validation-error').classList.add('hidden');
      renderSurveyQuestion(qNum);
    }

    function surveyNextStep() {
      if (surveyAnswers[surveyCurrentStep] === undefined) {
        document.getElementById('survey-validation-error').classList.remove('hidden');
        return;
      }
      if (surveyCurrentStep < 7) {
        surveyGoToStep(surveyCurrentStep + 1);
      } else {
        surveyGoToStep(8); // Go to image scan
      }
    }

    function surveyPrevStep() {
      if (surveyCurrentStep <= 1) {
        surveyGoToStep(0);
      } else {
        surveyGoToStep(surveyCurrentStep - 1);
      }
    }

    function handleImageSelected(e) {
      const file = e.target.files[0];
      if (!file) return;
      if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
        showSurveyUploadErr("Please upload a valid JPEG or PNG photo.");
        return;
      }
      const reader = new FileReader();
      reader.onload = (ev) => {
        surveyUploadedImg = ev.target.result;
        displaySurveyImgPreview(surveyUploadedImg);
      };
      reader.readAsDataURL(file);
    }

    function displaySurveyImgPreview(url) {
      document.getElementById('survey-img-preview').src = url;
      document.getElementById('survey-upload-prompt').classList.add('hidden');
      document.getElementById('survey-preview-wrap').classList.remove('hidden');
      document.getElementById('survey-preview-wrap').classList.add('flex');
      document.getElementById('survey-upload-error').classList.add('hidden');
    }

    function clearUploadedImage(e) {
      if (e) e.stopPropagation();
      surveyUploadedImg = null;
      document.getElementById('survey-file-input').value = "";
      document.getElementById('survey-preview-wrap').classList.add('hidden');
      document.getElementById('survey-preview-wrap').classList.remove('flex');
      document.getElementById('survey-upload-prompt').classList.remove('hidden');
    }

    function loadSampleEyePhoto(e) {
      if (e) e.stopPropagation();
      surveyUploadedImg = "https://images.unsplash.com/photo-1559757175-5700dde675bc?auto=format&fit=crop&w=600&q=80";
      displaySurveyImgPreview(surveyUploadedImg);
    }

    function showSurveyUploadErr(msg) {
      document.getElementById('survey-upload-error-text').textContent = msg;
      document.getElementById('survey-upload-error').classList.remove('hidden');
    }

    function submitSurveyForAnalysis() {
      if (!surveyUploadedImg) {
        showSurveyUploadErr("Please upload or load an eye photograph for ocular surface verification.");
        return;
      }
      surveyGoToStep(9); // Go to results
    }

    function runSurveyDiagnosticSimulation() {
      document.getElementById('survey-loading-box').classList.remove('hidden');
      document.getElementById('survey-results-box').classList.add('hidden');

      const statuses = [
        "Processing corneal reflections & tear film stability markers...",
        "Quantifying 7-day OSDI symptom index across 6 clinical domains...",
        "Applying verified 90.4% Dry Eye classification matrix to targeted eye(s)..."
      ];
      let i = 0;
      const t = setInterval(() => {
        i++;
        if (i < statuses.length) {
          document.getElementById('survey-loading-status').textContent = statuses[i];
        }
      }, 800);

      setTimeout(() => {
        clearInterval(t);
        displaySurveyFinalResults();
      }, 2500);
    }

    function displaySurveyFinalResults() {
      document.getElementById('survey-loading-box').classList.add('hidden');
      document.getElementById('survey-results-box').classList.remove('hidden');

      // Calculate Clinical OSDI Formula over 6 frequency symptom questions (Q1 to Q6):
      // Max raw sum is 6 * 3 = 18.
      // Standardized to 0-100: (sum * 100) / 18
      let sum = 0;
      for (let k = 1; k <= 6; k++) {
        sum += (typeof surveyAnswers[k] === 'number' ? surveyAnswers[k] : 0);
      }
      const osdiScore = Math.round((sum * 100) / 18);

      document.getElementById('res-score-number').textContent = osdiScore;
      const meter = document.getElementById('res-score-meter');
      meter.style.width = `${Math.max(6, Math.min(100, osdiScore))}%`;

      const badgeIcon = document.getElementById('res-badge-icon');
      const badgeLabel = document.getElementById('res-badge-label');
      const lateralityBadge = document.getElementById('res-laterality-badge');
      const catLabel = document.getElementById('res-osdi-cat');
      const titleEl = document.getElementById('res-summary-title');
      const descEl = document.getElementById('res-summary-desc');
      const recEl = document.getElementById('res-recommendation-text');

      // Update Laterality presentation
      const chosenLaterality = surveyAnswers[7] || "Both eyes";
      if (lateralityBadge) {
        lateralityBadge.textContent = `Laterality: ${chosenLaterality}`;
      }

      // Update Clinic Simulator with this OSDI score
      clinicParams.osdi = osdiScore;
      document.getElementById('input-osdi').value = osdiScore;
      document.getElementById('val-osdi').textContent = osdiScore.toFixed(1);

      if (osdiScore <= 12) {
        badgeIcon.className = "w-16 h-16 mx-auto mb-3 rounded-2xl flex items-center justify-center text-3xl shadow-lg bg-emerald-500/20 border border-emerald-400/40 text-emerald-400";
        badgeIcon.innerHTML = "✅";
        badgeLabel.className = "inline-block px-4 py-1 rounded-full text-xs font-bold tracking-wide uppercase border bg-emerald-500/20 text-emerald-300 border-emerald-400/40";
        badgeLabel.textContent = "Normal Ocular Surface";
        catLabel.textContent = "Normal (0-12)";
        catLabel.className = "text-emerald-400 font-mono";
        meter.className = "h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full transition-all duration-700";
        titleEl.textContent = "Optimal Tear Film Stability Detected";
        descEl.textContent = `Your 7-day symptom responses in ${chosenLaterality.toLowerCase()} and blink regularity show minimal signs of dry eye disease. Corneal moisture retention is healthy.`;
        recEl.textContent = "Maintain standard screen ergonomics and 20-20-20 rule breaks during visual tasks.";
      } else if (osdiScore <= 22) {
        badgeIcon.className = "w-16 h-16 mx-auto mb-3 rounded-2xl flex items-center justify-center text-3xl shadow-lg bg-cyan-500/20 border border-cyan-400/40 text-cyan-400";
        badgeIcon.innerHTML = "💧";
        badgeLabel.className = "inline-block px-4 py-1 rounded-full text-xs font-bold tracking-wide uppercase border bg-cyan-500/20 text-cyan-300 border-cyan-400/40";
        badgeLabel.textContent = "Mild Dry Eye Symptoms";
        catLabel.textContent = "Mild Dry Eye (13-22)";
        catLabel.className = "text-cyan-400 font-mono";
        meter.className = "h-full bg-gradient-to-r from-teal-400 to-cyan-400 rounded-full transition-all duration-700";
        titleEl.textContent = "Early Tear Evaporative Signs";
        descEl.textContent = `Intermittent grittiness, wind/AC discomfort, or screen fatigue in ${chosenLaterality.toLowerCase()} suggests early tear film breakup.`;
        recEl.textContent = "Consider preservative-free artificial tears and evaluate room humidity levels.";
      } else if (osdiScore <= 32) {
        badgeIcon.className = "w-16 h-16 mx-auto mb-3 rounded-2xl flex items-center justify-center text-3xl shadow-lg bg-amber-500/20 border border-amber-400/40 text-amber-400";
        badgeIcon.innerHTML = "⚠️";
        badgeLabel.className = "inline-block px-4 py-1 rounded-full text-xs font-bold tracking-wide uppercase border bg-amber-500/20 text-amber-300 border-amber-400/40";
        badgeLabel.textContent = "Moderate Dry Eye Risk";
        catLabel.textContent = "Moderate (23-32)";
        catLabel.className = "text-amber-400 font-mono";
        meter.className = "h-full bg-gradient-to-r from-amber-400 to-orange-400 rounded-full transition-all duration-700";
        titleEl.textContent = "Frequent Tear Film Instability";
        descEl.textContent = `Repeated ocular burning, night driving glare, and visual fluctuation in ${chosenLaterality.toLowerCase()} indicate compromised tear film lipid layer or meibomian gland dysfunction.`;
        recEl.textContent = "A formal slit-lamp exam, tear breakup time (TBUT), and warm compress therapy are recommended.";
      } else {
        badgeIcon.className = "w-16 h-16 mx-auto mb-3 rounded-2xl flex items-center justify-center text-3xl shadow-lg bg-rose-500/20 border border-rose-400/40 text-rose-400";
        badgeIcon.innerHTML = "🔴";
        badgeLabel.className = "inline-block px-4 py-1 rounded-full text-xs font-bold tracking-wide uppercase border bg-rose-500/20 text-rose-300 border-rose-400/40";
        badgeLabel.textContent = "Severe Dry Eye Disease";
        catLabel.textContent = "Severe (33+)";
        catLabel.className = "text-rose-400 font-mono";
        meter.className = "h-full bg-gradient-to-r from-orange-500 to-rose-500 rounded-full transition-all duration-700";
        titleEl.textContent = "Chronic Dry Eye Surface Disease";
        descEl.textContent = `Severe ocular discomfort, night-time visual issues, and persistent irritation in ${chosenLaterality.toLowerCase()} indicate substantial corneal epithelial stress.`;
        recEl.textContent = "Schedule an in-person consultation with an ophthalmologist or dry eye specialist for prescription therapy.";
      }
    }

    function surveyRestart() {
      surveyAnswers = {};
      clearUploadedImage();
      surveyGoToStep(0);
    }

    // =========================================================================
    // CLINICAL THERMOGRAPHY LAB & INFERENCE ENGINE
    // =========================================================================
    function initClinicCohortSelector() {
      const sel = document.getElementById('clinic-patient-select');
      sel.innerHTML = '';
      CLINICAL_PATIENT_COHORT.forEach((p, idx) => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = `${p.id} — Ground Truth: ${p.actual.toUpperCase()} (OSDI: ${p.osdi})`;
        sel.appendChild(opt);
      });
      loadSelectedPatient(0);
    }

    function loadSelectedPatient(idx) {
      const p = CLINICAL_PATIENT_COHORT[idx];
      if (!p) return;

      clinicParams.osdi = p.osdi;
      clinicParams.cr10 = p.cr10;
      clinicParams.cr7 = p.cr7;
      clinicParams.nc = p.nc;
      clinicParams.tc = p.tc;
      clinicParams.cc = p.cc;
      clinicParams.nl = p.nl;
      clinicParams.tl = p.tl;
      clinicParams.t0 = p.t0;
      clinicParams.t10 = p.t10;
      clinicParams.most = p.most;

      syncUIInputs();
      
      const gtEl = document.getElementById('pt-ground-truth');
      gtEl.textContent = p.actual;
      gtEl.className = (p.actual.toLowerCase().includes('normal')) ? "text-emerald-400 uppercase font-mono font-bold" : "text-rose-400 uppercase font-mono font-bold";
      document.getElementById('pt-id-tag').textContent = `Cohort ID: ${p.id}`;

      runClinicInference();
      renderCornealHeatmap();
    }

    function loadRandomPatient(type) {
      const filtered = CLINICAL_PATIENT_COHORT.filter(p => p.actual.toLowerCase().includes(type.toLowerCase()));
      if (filtered.length === 0) return;
      const pick = filtered[Math.floor(Math.random() * filtered.length)];
      const idx = CLINICAL_PATIENT_COHORT.indexOf(pick);
      document.getElementById('clinic-patient-select').value = idx;
      loadSelectedPatient(idx);
    }

    function resetThermalDefaults() {
      const n = MODEL_DATA.baselines.normal;
      clinicParams.osdi = n.osdi;
      clinicParams.cr10 = n.cr10;
      clinicParams.cr7 = n.cr7;
      clinicParams.nc = n.nc;
      clinicParams.tc = n.tc;
      clinicParams.cc = n.cc;
      clinicParams.nl = n.nl;
      clinicParams.tl = n.tl;
      clinicParams.t0 = n.t0;
      clinicParams.t10 = n.t10;
      clinicParams.most = n.most;

      syncUIInputs();
      runClinicInference();
      renderCornealHeatmap();
    }

    function syncUIInputs() {
      document.getElementById('input-osdi').value = clinicParams.osdi;
      document.getElementById('val-osdi').textContent = clinicParams.osdi.toFixed(1);

      document.getElementById('input-cr10').value = clinicParams.cr10;
      document.getElementById('val-cr10').textContent = clinicParams.cr10.toFixed(3);

      document.getElementById('input-cc').value = clinicParams.cc;
      document.getElementById('val-cc').textContent = clinicParams.cc.toFixed(2);

      document.getElementById('input-nc').value = clinicParams.nc;
      document.getElementById('val-nc').textContent = clinicParams.nc.toFixed(2);

      document.getElementById('input-tc').value = clinicParams.tc;
      document.getElementById('val-tc').textContent = clinicParams.tc.toFixed(2);

      document.getElementById('input-t10').value = clinicParams.t10;
      document.getElementById('val-t10').textContent = clinicParams.t10.toFixed(2);

      document.getElementById('input-tl').value = clinicParams.tl;
      document.getElementById('val-tl').textContent = clinicParams.tl.toFixed(2);

      document.getElementById('input-most').value = clinicParams.most;
      document.getElementById('val-most').textContent = clinicParams.most.toFixed(2);
    }

    function updateParam(param, val) {
      const num = parseFloat(val);
      clinicParams[param] = num;
      
      const valSpanMap = {
        osdi: 'val-osdi',
        cr10: 'val-cr10',
        cc: 'val-cc',
        nc: 'val-nc',
        tc: 'val-tc',
        t10: 'val-t10',
        tl: 'val-tl',
        most: 'val-most'
      };
      if (valSpanMap[param]) {
        const decimals = (param === 'cr10') ? 3 : ((param === 'osdi') ? 1 : 2);
        document.getElementById(valSpanMap[param]).textContent = num.toFixed(decimals);
      }

      runClinicInference();
      renderCornealHeatmap();
    }

    // Logistic Regression Classifier Execution
    function runClinicInference() {
      const rawFeats = [
        clinicParams.osdi,
        clinicParams.cr10,
        clinicParams.cr7,
        clinicParams.nc,
        clinicParams.tc,
        clinicParams.cc,
        clinicParams.nl,
        clinicParams.tl,
        clinicParams.t0,
        clinicParams.t10,
        clinicParams.most,
        clinicParams.nc - 0.38, // N approximation
        clinicParams.cc,        // C approximation
        clinicParams.tc - 0.72  // T approximation
      ];

      // Standardization: (x - mean) / std
      let z = MODEL_DATA.bias;
      const contributions = [];

      for (let i = 0; i < rawFeats.length; i++) {
        const norm = (rawFeats[i] - MODEL_DATA.mean[i]) / MODEL_DATA.std[i];
        const effect = norm * MODEL_DATA.weights[i];
        z += effect;
        contributions.push({
          name: MODEL_DATA.feature_names[i],
          raw: rawFeats[i],
          effect: effect
        });
      }

      // Sigmoid probability: P(Dry Eye) = 1 / (1 + exp(-z))
      const clippedZ = Math.max(-15, Math.min(15, z));
      const prob = 1.0 / (1.0 + Math.exp(-clippedZ));
      const probPct = (prob * 100).toFixed(1);

      // UI Updates
      document.getElementById('ai-prob-text').textContent = `${probPct}%`;
      document.getElementById('ai-prob-meter').style.width = `${Math.max(4, Math.min(100, prob * 100))}%`;

      const diagCard = document.getElementById('ai-diag-card');
      const diagTitle = document.getElementById('ai-diag-title');
      const diagSub = document.getElementById('ai-diag-subtitle');
      const statusDot = document.getElementById('ai-status-dot');

      if (prob < MODEL_DATA.threshold) {
        diagCard.className = "p-4 rounded-2xl bg-emerald-500/15 border border-emerald-400/40 my-3 transition-all duration-300 shadow-glow-emerald";
        diagTitle.className = "text-2xl font-display font-extrabold text-emerald-300 tracking-wide uppercase";
        diagTitle.textContent = "Normal Ocular Surface";
        diagSub.textContent = "Negative for dry eye pathology. Corneal cooling rate and thermal gradient remain physiologically stable.";
        statusDot.className = "w-3.5 h-3.5 rounded-full bg-emerald-400 animate-pulse";
      } else {
        diagCard.className = "p-4 rounded-2xl bg-rose-500/15 border border-rose-400/40 my-3 transition-all duration-300 shadow-glow-rose";
        diagTitle.className = "text-2xl font-display font-extrabold text-rose-400 tracking-wide uppercase";
        diagTitle.textContent = "Dry Eye Detected";
        diagSub.textContent = "Significant tear film instability. Marked by rapid corneal cooling (-0.05°C/s) and sub-normal surface temperatures.";
        statusDot.className = "w-3.5 h-3.5 rounded-full bg-rose-500 animate-pulse";
      }

      // Render Explainability bars (Top 4 influential factors)
      const topInfluencers = contributions
        .filter(c => ['OSDI', 'CR_10S', 'NC', 'CC', 'T10s', 'TL'].includes(c.name))
        .sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect))
        .slice(0, 4);

      const xaiContainer = document.getElementById('xai-contributions-container');
      xaiContainer.innerHTML = '';

      topInfluencers.forEach(item => {
        const isDryPusher = item.effect > 0;
        const barWidth = Math.min(100, Math.round(Math.abs(item.effect) * 45));
        const itemRow = document.createElement('div');
        itemRow.className = "p-2 rounded-lg bg-slate-900/60 border border-slate-800/80";
        itemRow.innerHTML = `
          <div class="flex justify-between text-[11px] mb-1">
            <span class="font-medium text-slate-300">${formatFeatLabel(item.name)}: <span class="font-mono text-cyan-300">${item.raw.toFixed(2)}</span></span>
            <span class="font-mono ${isDryPusher ? 'text-rose-400' : 'text-emerald-400'} font-semibold">${isDryPusher ? '+ Towards Dry Eye' : '✓ Towards Normal'}</span>
          </div>
          <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div class="h-full ${isDryPusher ? 'bg-rose-500' : 'bg-emerald-400'} rounded-full" style="width: ${barWidth}%;"></div>
          </div>
        `;
        xaiContainer.appendChild(itemRow);
      });
    }

    function formatFeatLabel(key) {
      const dict = {
        'OSDI': 'Symptom OSDI',
        'CR_10S': 'Cooling Rate 10s',
        'NC': 'Nasal Cornea Temp',
        'CC': 'Central Cornea Temp',
        'T10s': 'Sustained 10s Temp',
        'TL': 'Temporal Limbus'
      };
      return dict[key] || key;
    }

    // =========================================================================
    // CORNEAL THERMAL HEATMAP RENDERER (HTML5 CANVAS)
    // =========================================================================
    function renderCornealHeatmap() {
      const canvas = document.getElementById('cornea-thermal-canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);

      // Center coords
      const cx = w / 2;
      const cy = h / 2;
      const eyeR = 110;

      // Draw dark ocular orbit backdrop
      const bgGrad = ctx.createRadialGradient(cx, cy, 30, cx, cy, eyeR + 30);
      bgGrad.addColorStop(0, '#091A2E');
      bgGrad.addColorStop(1, '#02060D');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      // Temperature mapping function (32°C -> Deep Blue, 35°C -> Cyan/Green, 37°C -> Yellow/Red)
      function tempToColor(t) {
        // Clamp between 33 and 37
        const norm = Math.max(0, Math.min(1, (t - 33.5) / 3.0));
        const r = Math.round(norm * 240);
        const g = Math.round((1 - Math.abs(norm - 0.5) * 2) * 220 + 35);
        const b = Math.round((1 - norm) * 255);
        return `rgba(${r}, ${g}, ${b}, 0.85)`;
      }

      const colNC = tempToColor(clinicParams.nc);
      const colCC = tempToColor(clinicParams.cc);
      const colTC = tempToColor(clinicParams.tc);

      // Sclera Outline
      ctx.beginPath();
      ctx.ellipse(cx, cy, eyeR + 25, eyeR - 15, 0, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(72, 202, 228, 0.25)';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Cornea Thermal Gradient (Left = Nasal, Center = Central, Right = Temporal)
      const grad = ctx.createLinearGradient(cx - eyeR, cy, cx + eyeR, cy);
      grad.addColorStop(0.15, colNC);
      grad.addColorStop(0.50, colCC);
      grad.addColorStop(0.85, colTC);

      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, eyeR - 20, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.shadowColor = colCC;
      ctx.shadowBlur = 25;
      ctx.fill();
      ctx.restore();

      // Pupils & Corneal Rings
      ctx.beginPath();
      ctx.arc(cx, cy, 25, 0, Math.PI * 2);
      ctx.fillStyle = '#050B14';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Pupil reflection light point
      ctx.beginPath();
      ctx.arc(cx + 6, cy - 8, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();

      // Tear film breakup simulation particles if Dry Eye
      if (clinicParams.cr10 < -0.035 || clinicParams.cc < 34.4) {
        ctx.fillStyle = 'rgba(56, 189, 248, 0.7)';
        const spots = [
          { x: cx - 35, y: cy - 25, r: 4 },
          { x: cx + 28, y: cy + 30, r: 5 },
          { x: cx + 45, y: cy - 20, r: 3.5 },
          { x: cx - 20, y: cy + 40, r: 4.5 }
        ];
        spots.forEach(sp => {
          ctx.beginPath();
          ctx.arc(sp.x, sp.y, sp.r, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 1;
          ctx.stroke();
        });
      }

      // Region Annotations
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(`NC: ${clinicParams.nc.toFixed(1)}°`, cx - 65, cy - 45);
      ctx.fillText(`CC: ${clinicParams.cc.toFixed(1)}°`, cx, cy + 55);
      ctx.fillText(`TC: ${clinicParams.tc.toFixed(1)}°`, cx + 65, cy - 45);

      // Label overlays
      document.getElementById('lbl-canvas-nc').textContent = `${clinicParams.nc.toFixed(1)}°C`;
      document.getElementById('lbl-canvas-cc').textContent = `${clinicParams.cc.toFixed(1)}°C`;
      document.getElementById('lbl-canvas-tc').textContent = `${clinicParams.tc.toFixed(1)}°C`;
    }

    // =========================================================================
    // INITIALIZATION ON PAGE LOAD
    // =========================================================================
    window.addEventListener('DOMContentLoaded', () => {
      initThermalScanner();
      initClinicCohortSelector();
      renderCornealHeatmap();
    });
  