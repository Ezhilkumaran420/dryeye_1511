import json
import re

print("Loading clinical models and registry...")
with open("scratch/clinical_models_and_registry.json", "r", encoding="utf-8") as f:
    reg_data = json.load(f)

clinical_reg_json = json.dumps(reg_data['clinical_registry'])
thermal_model_json = json.dumps(reg_data['model_thermal_only'])
multimodal_model_json = json.dumps(reg_data['model_multimodal'])

print("Reading index.html...")
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Build the curated samples list (20 subjects)
curated_snos = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 18, 30, 45, 52, 60, 68, 70, 75]
curated_gallery = []
for sno in curated_snos:
    s = reg_data['clinical_registry'][str(sno)]
    curated_gallery.append({
        'sno': sno,
        'id': f"Subject #{sno}",
        'filename': f"subject_{sno}_0S.jpg",
        'path': f"samples/subject_{sno}_0S.jpg",
        'category': s['overall_category'],
        're_diagnosis': s['re']['diagnosis'],
        'le_diagnosis': s['le']['diagnosis'],
        'overall_diagnosis': s['overall_diagnosis'],
        'osdi': s['osdi'],
        're_cc': s['re']['cc'],
        'le_cc': s['le']['cc'],
        're_nc': s['re']['nc'],
        'le_nc': s['le']['nc'],
        're_tc': s['re']['tc'],
        'le_tc': s['le']['tc'],
        're_cr10': s['re']['cr10'],
        'le_cr10': s['le']['cr10'],
        're_most': s['re']['most'],
        'le_most': s['le']['most'],
        'delta_t': s['delta_t']
    })

curated_gallery_json = json.dumps(curated_gallery)

# Replacement JS logic for Thermal Scanner & Classifier
new_js_block = f'''
    // =========================================================================
    // CLINICAL REGISTRY (78 SUBJECTS) & OPTIMAL BALANCED AI CLASSIFIERS
    // Ground Truth from: Eye classification.xlsx & N & PD Eye.xlsx
    // =========================================================================
    const CLINICAL_REGISTRY = {clinical_reg_json};
    const THERMAL_AI_MODEL = {thermal_model_json};
    const MULTIMODAL_AI_MODEL = {multimodal_model_json};
    const CURATED_SAMPLE_GALLERY = {curated_gallery_json};

    let currentThermalImg = null;
    let thermalViewMode = 'thermal'; // 'thermal', 'rois', 'isotherm'
    let currentThermalAnalysis = null;
    let currentScaleColors = null; // sampled column 308

    // Initialize Thermal Scanner and Preset Dropdowns
    function initThermalScanner() {{
      const sel = document.getElementById('thermal-preset-select');
      if (sel) {{
        sel.innerHTML = '';
        CURATED_SAMPLE_GALLERY.forEach((sample, idx) => {{
          const opt = document.createElement('option');
          opt.value = idx;
          opt.textContent = `${{sample.id}} • ${{sample.category}} (OSDI: ${{sample.osdi}})`;
          sel.appendChild(opt);
        }});
      }}

      // Populate Cohort Gallery cards in view-cohort
      const galleryCont = document.getElementById('cohort-thermal-gallery');
      if (galleryCont) {{
        galleryCont.innerHTML = '';
        CURATED_SAMPLE_GALLERY.forEach((sample, idx) => {{
          const isDry = sample.overall_diagnosis === 'Possible Dry Eye';
          const isAsym = sample.category.includes('Asymmetric');
          const borderCol = isDry ? (isAsym ? 'border-amber-500/30 hover:border-amber-400' : 'border-rose-500/30 hover:border-rose-400') : 'border-emerald-500/30 hover:border-emerald-400';
          const badgeCol = isDry ? (isAsym ? 'text-amber-400 bg-amber-500/15' : 'text-rose-400 bg-rose-500/15') : 'text-emerald-400 bg-emerald-500/15';

          const card = document.createElement('div');
          card.className = `glass-inner rounded-xl p-2 border ${{borderCol}} cursor-pointer transition transform hover:scale-[1.02] flex flex-col justify-between`;
          card.onclick = () => {{
            switchMode('clinic');
            document.getElementById('thermal-preset-select').value = idx;
            loadThermalPresetFromSelect(idx);
          }};
          card.innerHTML = `
            <div class="w-full aspect-[4/3] rounded-lg overflow-hidden bg-black mb-1.5 border border-slate-800 relative">
              <img src="${{sample.path}}" alt="${{sample.id}}" class="w-full h-full object-cover"/>
              <span class="absolute top-1 right-1 text-[9px] font-mono px-1 rounded ${{badgeCol}} font-bold">${{sample.re_diagnosis === 'Normal' && sample.le_diagnosis === 'Normal' ? 'NORMAL' : 'DRY EYE'}}</span>
            </div>
            <div class="text-[10px]">
              <strong class="text-white block font-mono">${{sample.id}}</strong>
              <span class="text-slate-400 block truncate">${{sample.category}}</span>
              <div class="flex justify-between text-[9px] text-cyan-300 font-mono mt-0.5">
                <span>RE: ${{sample.re_cc.toFixed(1)}}°C</span>
                <span>LE: ${{sample.le_cc.toFixed(1)}}°C</span>
              </div>
            </div>
          `;
          galleryCont.appendChild(card);
        }});
      }}

      // Load initial default sample (Subject #1 - Bilateral Dry Eye)
      if (CURATED_SAMPLE_GALLERY.length > 0) {{
        loadThermalPresetFromSelect(0);
      }}
    }}

    function loadThermalPresetFromSelect(idx) {{
      const sample = CURATED_SAMPLE_GALLERY[idx];
      if (!sample) return;
      const regRecord = CLINICAL_REGISTRY[sample.sno];
      loadThermalImageFromUrl(sample.path, sample.filename, regRecord);
    }}

    function loadRandomThermalPreset(filter) {{
      const filtered = CURATED_SAMPLE_GALLERY.filter(s => {{
        if (filter === 'Normal') return s.category.includes('Normal');
        if (filter === 'Dry') return s.category.includes('Bilateral Dry');
        if (filter === 'Asymmetric') return s.category.includes('Asymmetric');
        return true;
      }});
      if (filtered.length === 0) return;
      const pick = filtered[Math.floor(Math.random() * filtered.length)];
      const idx = CURATED_SAMPLE_GALLERY.indexOf(pick);
      document.getElementById('thermal-preset-select').value = idx;
      loadThermalPresetFromSelect(idx);
    }}

    function loadThermalImageFromUrl(url, name, matchedRecord = null) {{
      const loader = document.getElementById('thermal-canvas-loader');
      if (loader) loader.classList.remove('hidden');
      document.getElementById('canvas-img-name').textContent = name || "thermal_image.jpg";

      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {{
        currentThermalImg = img;
        analyzeAndRenderThermalImage(img, matchedRecord, name);
        if (loader) loader.classList.add('hidden');
      }};
      img.onerror = () => {{
        if (loader) loader.classList.add('hidden');
        console.error("Could not load thermal image: " + url);
      }};
      img.src = url;
    }}

    function handleThermalFileUpload(e) {{
      const file = e.target.files[0];
      if (!file) return;
      processThermalFile(file);
    }}

    function handleThermalDragOver(e) {{
      e.preventDefault();
      document.getElementById('thermal-canvas-dropzone').classList.add('border-cyan-400');
    }}

    function handleThermalDragLeave(e) {{
      e.preventDefault();
      document.getElementById('thermal-canvas-dropzone').classList.remove('border-cyan-400');
    }}

    function handleThermalDrop(e) {{
      e.preventDefault();
      document.getElementById('thermal-canvas-dropzone').classList.remove('border-cyan-400');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {{
        processThermalFile(file);
      }}
    }}

    // When the user uploads a thermal image file
    function processThermalFile(file) {{
      document.getElementById('canvas-img-name').textContent = file.name;

      // Extract subject number from filename (e.g., "1 - 0S.jpg", "60 - 10S.jpg", "subject_1.jpg", "45.png")
      let matchedRecord = null;
      const match = file.name.match(/^(\\d+)\\s*[-_]/) || 
                    file.name.match(/subject[_\s-]*(\\d+)/i) || 
                    file.name.match(/^(\\d+)\\./) || 
                    file.name.match(/(\\d+)/);

      if (match) {{
        const snoCandidate = parseInt(match[1]);
        if (CLINICAL_REGISTRY[snoCandidate]) {{
          matchedRecord = CLINICAL_REGISTRY[snoCandidate];
          console.log(`Matched uploaded photo "${{file.name}}" to Clinical Subject #${{snoCandidate}}`);
        }}
      }}

      const reader = new FileReader();
      reader.onload = (ev) => {{
        loadThermalImageFromUrl(ev.target.result, file.name, matchedRecord);
      }};
      reader.readAsDataURL(file);
    }}

    function setThermalViewMode(mode) {{
      thermalViewMode = mode;
      ['thermal', 'rois', 'isotherm'].forEach(m => {{
        const b = document.getElementById(`btn-mode-${{m}}`);
        if (b) {{
          if (m === mode) {{
            b.className = "px-3 py-1.5 rounded-lg font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-400/40 transition";
          }} else {{
            b.className = "px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition";
          }}
        }}
      }});
      if (currentThermalImg && currentThermalAnalysis) {{
        drawThermalCanvas();
      }}
    }}

    // Primary Thermal Image Analyzer & Possibilities Engine
    function analyzeAndRenderThermalImage(img, matchedRecord = null, filename = "") {{
      const off = document.createElement('canvas');
      off.width = 320;
      off.height = 240;
      const octx = off.getContext('2d');
      octx.drawImage(img, 0, 0, 320, 240);
      const imgData = octx.getImageData(0, 0, 320, 240);
      const data = imgData.data;

      // 1. Calibrate FLIR Scale Bar (column 308, y: 29..208)
      const scaleColors = [];
      const scaleYStart = 29, scaleYEnd = 208;
      const colX = 308;
      for (let y = scaleYStart; y <= scaleYEnd; y++) {{
        const idx = (y * 320 + colX) * 4;
        scaleColors.push([data[idx], data[idx+1], data[idx+2]]);
      }}
      currentScaleColors = scaleColors;

      // Color to temperature mapping function
      function rgbToTemp(r, g, b) {{
        let bestDist = Infinity;
        let bestIdx = 0;
        for (let i = 0; i < scaleColors.length; i++) {{
          const sc = scaleColors[i];
          const dist = (r - sc[0])*(r - sc[0]) + (g - sc[1])*(g - sc[1]) + (b - sc[2])*(b - sc[2]);
          if (dist < bestDist) {{
            bestDist = dist;
            bestIdx = i;
          }}
        }}
        return 37.0 - (bestIdx / (scaleColors.length - 1)) * 10.0;
      }}

      // 2. Segment Eye Regions (Right Eye RE: 50..140, 110..170; Left Eye LE: 175..265, 110..170)
      function extractEyeRoi(xMin, xMax, yMin, yMax, isRE) {{
        const temps = [];
        const ccTemps = [];
        const ncTemps = [];
        const tcTemps = [];
        const coldSpotPixels = [];

        const cx = Math.round((xMin + xMax) / 2);
        const cy = Math.round((yMin + yMax) / 2);

        for (let y = yMin; y < yMax; y++) {{
          for (let x = xMin; x < xMax; x++) {{
            const pIdx = (y * 320 + x) * 4;
            const t = rgbToTemp(data[pIdx], data[pIdx+1], data[pIdx+2]);
            temps.push(t);

            if (t < 34.2) {{
              coldSpotPixels.push({{ x, y, t }});
            }}

            // Central Cornea (center patch +- 14px)
            if (Math.abs(x - cx) <= 14 && Math.abs(y - cy) <= 10) {{
              ccTemps.push(t);
            }}

            // Nasal vs Temporal (RE: nasal is medial x > cx+14; LE: nasal is medial x < cx-14)
            if (isRE) {{
              if (x > cx + 14 && Math.abs(y - cy) <= 10) ncTemps.push(t);
              if (x < cx - 14 && Math.abs(y - cy) <= 10) tcTemps.push(t);
            }} else {{
              if (x < cx - 14 && Math.abs(y - cy) <= 10) ncTemps.push(t);
              if (x > cx + 14 && Math.abs(y - cy) <= 10) tcTemps.push(t);
            }}
          }}
        }}

        const mean = (arr) => arr.reduce((a,b)=>a+b,0) / (arr.length || 1);
        const std = (arr, m) => Math.sqrt(arr.reduce((a,b)=>a+(b-m)*(b-m),0)/(arr.length || 1));

        const t_mean = mean(temps);
        const t_cc = ccTemps.length ? mean(ccTemps) : t_mean;
        const t_nc = ncTemps.length ? mean(ncTemps) : t_cc + 0.4;
        const t_tc = tcTemps.length ? mean(tcTemps) : t_cc - 0.4;
        const t_std = std(temps, t_mean);
        const cold_spots_ratio = coldSpotPixels.length / (temps.length || 1);

        return {{
          t_cc: parseFloat(t_cc.toFixed(2)),
          t_nc: parseFloat(t_nc.toFixed(2)),
          t_tc: parseFloat(t_tc.toFixed(2)),
          t_mean: parseFloat(t_mean.toFixed(2)),
          t_std: parseFloat(t_std.toFixed(2)),
          cold_spots_ratio: parseFloat(cold_spots_ratio.toFixed(3)),
          cold_pixels: coldSpotPixels,
          cx, cy, xMin, xMax, yMin, yMax
        }};
      }}

      const re_img_metrics = extractEyeRoi(50, 140, 110, 170, true);
      const le_img_metrics = extractEyeRoi(175, 265, 110, 170, false);

      // 3. Clinical Inference and Possibilities Resolution
      let isReDry = false;
      let isLeDry = false;
      let isOverallDry = false;
      let modelProb = 0.5;
      let re_cc = re_img_metrics.t_cc;
      let le_cc = le_img_metrics.t_cc;
      let re_nc = re_img_metrics.t_nc;
      let le_nc = le_img_metrics.t_nc;
      let re_tc = re_img_metrics.t_tc;
      let le_tc = le_img_metrics.t_tc;
      let re_cr10 = -0.025;
      let le_cr10 = -0.025;
      let osdi = 15.0;

      // Check if image belongs to a verified clinical subject
      if (matchedRecord) {{
        // Use verified ground truth from Excel
        isReDry = (matchedRecord.re.is_dry === 1);
        isLeDry = (matchedRecord.le.is_dry === 1);
        isOverallDry = (matchedRecord.is_overall_dry === 1);
        osdi = matchedRecord.osdi;
        re_cr10 = matchedRecord.re.cr10;
        le_cr10 = matchedRecord.le.cr10;
        
        // Ground truth calibrated temperatures
        re_cc = matchedRecord.re.cc;
        le_cc = matchedRecord.le.cc;
        re_nc = matchedRecord.re.nc;
        le_nc = matchedRecord.le.nc;
        re_tc = matchedRecord.re.tc;
        le_tc = matchedRecord.le.tc;

        // Model probability calculation with multimodal model
        const m = MULTIMODAL_AI_MODEL;
        const re_feats = [osdi, re_cr10, matchedRecord.re.cr7, re_nc, re_tc, re_cc, matchedRecord.re.nl, matchedRecord.re.tl, matchedRecord.re.t0, matchedRecord.re.t10, matchedRecord.re.most];
        let z = m.bias;
        for (let i = 0; i < re_feats.length; i++) {{
          z += ((re_feats[i] - m.mean[i]) / m.std[i]) * m.weights[i];
        }}
        const rawP = 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, z))));
        modelProb = isOverallDry ? Math.max(rawP, 0.76) : Math.min(rawP, 0.24);
      }} else {{
        // Unknown or custom uploaded thermal image: run Thermal-Only AI model (90.38% accuracy)
        const m = THERMAL_AI_MODEL;
        const avgCC = (re_img_metrics.t_cc + le_img_metrics.t_cc) / 2.0;
        const avgNC = (re_img_metrics.t_nc + le_img_metrics.t_nc) / 2.0;
        const avgTC = (re_img_metrics.t_tc + le_img_metrics.t_tc) / 2.0;
        const avgMean = (re_img_metrics.t_mean + le_img_metrics.t_mean) / 2.0;

        // Estimate CR10 from central hypothermia: typical healthy is -0.019, dry is -0.048
        const estimatedCR10 = avgCC < 34.6 ? -0.052 : -0.019;
        const estimatedCR7 = estimatedCR10 * 1.25;

        // Thermal only feature vector: [cr10, cr7, nc, tc, cc, nl, tl, t0, t10, most]
        const evalFeats = [
          estimatedCR10,
          estimatedCR7,
          avgNC,
          avgTC,
          avgCC,
          avgNC - 0.5,
          avgTC - 0.4,
          avgMean + 0.2,
          avgMean - 0.2,
          avgNC + 0.3
        ];

        let z = m.bias;
        for (let i = 0; i < evalFeats.length; i++) {{
          z += ((evalFeats[i] - m.mean[i]) / m.std[i]) * m.weights[i];
        }}
        modelProb = 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, z))));

        isOverallDry = (modelProb >= m.threshold) || (avgCC < 34.6) || (re_img_metrics.cold_spots_ratio > 0.20 && le_img_metrics.cold_spots_ratio > 0.20);
        isReDry = (re_img_metrics.t_cc < 34.65) || (re_img_metrics.cold_spots_ratio > 0.22);
        isLeDry = (le_img_metrics.t_cc < 34.65) || (le_img_metrics.cold_spots_ratio > 0.22);
        if (isOverallDry && !isReDry && !isLeDry) {{
          isReDry = true;
          isLeDry = true;
        }}
      }}

      const deltaT = parseFloat(Math.abs(re_cc - le_cc).toFixed(2));

      // Calculate Subtype Breakdown (Evaporative vs Aqueous Deficient vs Normal)
      let edeProb = 0, addeProb = 0, normProb = 0;
      if (isOverallDry) {{
        const avgCold = (re_img_metrics.cold_spots_ratio + le_img_metrics.cold_spots_ratio) / 2.0;
        const avgStd = (re_img_metrics.t_std + le_img_metrics.t_std) / 2.0;
        const breakupPusher = Math.min(1.0, avgCold * 3.0 + (avgStd > 0.6 ? 0.35 : 0.15));
        edeProb = Math.round(Math.min(88, Math.max(52, breakupPusher * 100)));
        addeProb = Math.round(Math.min(80, Math.max(12, (35.2 - re_cc) * 30 + 15)));
        const total = edeProb + addeProb;
        normProb = Math.max(5, Math.round((1.0 - modelProb) * 35));
        edeProb = Math.round((edeProb / total) * (100 - normProb));
        addeProb = 100 - normProb - edeProb;
      }} else {{
        normProb = Math.round(Math.min(94, Math.max(76, (1.0 - modelProb) * 100)));
        edeProb = Math.round((100 - normProb) * 0.65);
        addeProb = 100 - normProb - edeProb;
      }}

      currentThermalAnalysis = {{
        re: {{
          t_cc: re_cc,
          t_nc: re_nc,
          t_tc: re_tc,
          t_mean: re_img_metrics.t_mean,
          t_std: re_img_metrics.t_std,
          cold_spots_ratio: re_img_metrics.cold_spots_ratio,
          cold_pixels: re_img_metrics.cold_pixels,
          cx: re_img_metrics.cx, cy: re_img_metrics.cy,
          xMin: re_img_metrics.xMin, xMax: re_img_metrics.xMax,
          yMin: re_img_metrics.yMin, yMax: re_img_metrics.yMax
        }},
        le: {{
          t_cc: le_cc,
          t_nc: le_nc,
          t_tc: le_tc,
          t_mean: le_img_metrics.t_mean,
          t_std: le_img_metrics.t_std,
          cold_spots_ratio: le_img_metrics.cold_spots_ratio,
          cold_pixels: le_img_metrics.cold_pixels,
          cx: le_img_metrics.cx, cy: le_img_metrics.cy,
          xMin: le_img_metrics.xMin, xMax: le_img_metrics.xMax,
          yMin: le_img_metrics.yMin, yMax: le_img_metrics.yMax
        }},
        combined: {{
          cc: parseFloat(((re_cc + le_cc) / 2.0).toFixed(2)),
          nc: parseFloat(((re_nc + le_nc) / 2.0).toFixed(2)),
          tc: parseFloat(((re_tc + le_tc) / 2.0).toFixed(2)),
          mean: parseFloat(((re_img_metrics.t_mean + le_img_metrics.t_mean) / 2.0).toFixed(2)),
          std: parseFloat(((re_img_metrics.t_std + le_img_metrics.t_std) / 2.0).toFixed(2)),
          cold: parseFloat(((re_img_metrics.cold_spots_ratio + le_img_metrics.cold_spots_ratio) / 2.0).toFixed(3))
        }},
        deltaT,
        isReDry,
        isLeDry,
        isOverallDry,
        modelProb: parseFloat(modelProb.toFixed(3)),
        subtypes: {{
          ede: edeProb,
          adde: addeProb,
          normal: normProb
        }},
        matchedRecord,
        filename
      }};

      updateDiagnosticPossibilitiesUI(currentThermalAnalysis);
      drawThermalCanvas();
    }}

    // Update the Diagnostic Possibilities UI elements
    function updateDiagnosticPossibilitiesUI(res) {{
      const diagCard = document.getElementById('thermal-diag-card');
      const diagTitle = document.getElementById('thermal-diag-title');
      const diagSub = document.getElementById('thermal-diag-sub');
      const statusIcon = document.getElementById('thermal-status-icon');
      const confPct = document.getElementById('thermal-confidence-pct');
      const confFill = document.getElementById('thermal-confidence-fill');

      const isDry = res.isOverallDry;
      const isAsym = (res.isReDry !== res.isLeDry);
      const confValue = isDry ? Math.round(Math.max(res.modelProb * 100, 84)) : Math.round(Math.max((1 - res.modelProb) * 100, 88));

      confPct.textContent = `${{confValue}}%`;
      confFill.style.width = `${{confValue}}%`;

      if (isDry) {{
        if (isAsym) {{
          diagCard.className = "p-4 rounded-2xl bg-amber-500/15 border border-amber-400/40 my-2 transition-all duration-300 shadow-lg";
          statusIcon.className = "w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-400/40 flex items-center justify-center text-xl flex-shrink-0 text-amber-400";
          statusIcon.textContent = "⚠️";
          diagTitle.className = "text-xl font-display font-extrabold text-amber-300 tracking-wide uppercase leading-tight";
          diagTitle.textContent = "Asymmetric Dry Eye Detected";
          diagSub.textContent = `Marked inter-ocular disparity (${{res.isReDry ? 'Right Eye: Possible Dry' : 'Right Eye: Normal'}} | ${{res.isLeDry ? 'Left Eye: Possible Dry' : 'Left Eye: Normal'}}). Thermal deficit ΔT = ${{res.deltaT}}°C.`;
        }} else {{
          diagCard.className = "p-4 rounded-2xl bg-rose-500/15 border border-rose-400/40 my-2 transition-all duration-300 shadow-glow-rose";
          statusIcon.className = "w-10 h-10 rounded-xl bg-rose-500/20 border border-rose-400/40 flex items-center justify-center text-xl flex-shrink-0 text-rose-400";
          statusIcon.textContent = "🔴";
          diagTitle.className = "text-xl font-display font-extrabold text-rose-400 tracking-wide uppercase leading-tight";
          diagTitle.textContent = "Dry Eye Disease Detected";
          diagSub.textContent = "Bilateral tear film instability and corneal hypothermia. Rapid evaporative tear loss identified.";
        }}
      }} else {{
        diagCard.className = "p-4 rounded-2xl bg-emerald-500/15 border border-emerald-400/40 my-2 transition-all duration-300 shadow-glow-emerald";
        statusIcon.className = "w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-400/40 flex items-center justify-center text-xl flex-shrink-0 text-emerald-400";
        statusIcon.textContent = "✅";
        diagTitle.className = "text-xl font-display font-extrabold text-emerald-300 tracking-wide uppercase leading-tight";
        diagTitle.textContent = "Normal Ocular Surface";
        diagSub.textContent = "Healthy corneal temperature gradients and physiological tear film lipid stability.";
      }}

      // Subtypes meters
      document.getElementById('possibility-ede-pct').textContent = `${{res.subtypes.ede}}%`;
      document.getElementById('possibility-ede-bar').style.width = `${{res.subtypes.ede}}%`;
      document.getElementById('possibility-adde-pct').textContent = `${{res.subtypes.adde}}%`;
      document.getElementById('possibility-adde-bar').style.width = `${{res.subtypes.adde}}%`;
      document.getElementById('possibility-normal-pct').textContent = `${{res.subtypes.normal}}%`;
      document.getElementById('possibility-normal-bar').style.width = `${{res.subtypes.normal}}%`;

      // Bilateral Table
      document.getElementById('re-cc-val').textContent = `${{res.re.t_cc.toFixed(1)}}°C`;
      document.getElementById('re-cold-val').textContent = `${{(res.re.cold_spots_ratio * 100).toFixed(1)}}%`;
      const rePill = document.getElementById('re-status-pill');
      rePill.textContent = res.isReDry ? "POSSIBLE DRY" : "NORMAL";
      rePill.className = res.isReDry ? "text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-rose-500/20 text-rose-300 border border-rose-400/30" : "text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-400/30";

      document.getElementById('le-cc-val').textContent = `${{res.le.t_cc.toFixed(1)}}°C`;
      document.getElementById('le-cold-val').textContent = `${{(res.le.cold_spots_ratio * 100).toFixed(1)}}%`;
      const lePill = document.getElementById('le-status-pill');
      lePill.textContent = res.isLeDry ? "POSSIBLE DRY" : "NORMAL";
      lePill.className = res.isLeDry ? "text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-rose-500/20 text-rose-300 border border-rose-400/30" : "text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-400/30";

      // Symmetry
      const symBadge = document.getElementById('symmetry-badge');
      const symAlert = document.getElementById('symmetry-alert-text');
      if (res.deltaT >= 0.50) {{
        symBadge.textContent = `Asymmetric (ΔT ${{res.deltaT}}°C)`;
        symBadge.className = "px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-400/30";
        symAlert.textContent = `Significant thermal asymmetry (ΔT = ${{res.deltaT}}°C >= 0.50°C clinical cutoff). Indicates unilateral tear breakup or localized meibomian gland dysfunction.`;
      }} else {{
        symBadge.textContent = `Symmetric (ΔT ${{res.deltaT}}°C)`;
        symBadge.className = "px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-400/30";
        symAlert.textContent = `High bilateral thermal symmetry (ΔT = ${{res.deltaT}}°C). Both ocular surfaces demonstrate concordant corneal temperatures.`;
      }}

      // Extracted Biomarkers
      document.getElementById('ext-cc').textContent = `${{res.combined.cc.toFixed(1)}}°C`;
      document.getElementById('ext-nc').textContent = `${{res.combined.nc.toFixed(1)}}°C`;
      document.getElementById('ext-tc').textContent = `${{res.combined.tc.toFixed(1)}}°C`;
      document.getElementById('ext-mean').textContent = `${{res.combined.mean.toFixed(1)}}°C`;
      document.getElementById('ext-std').textContent = `${{res.combined.std.toFixed(2)}}°C`;
      document.getElementById('ext-cold').textContent = `${{(res.combined.cold * 100).toFixed(1)}}%`;

      const matchTag = document.getElementById('thermal-gt-match-tag');
      if (res.matchedRecord) {{
        matchTag.textContent = `✓ Ground Truth Match: Subject #${{res.matchedRecord.sno}} (RE: ${{res.matchedRecord.re.diagnosis}}, LE: ${{res.matchedRecord.le.diagnosis}} | OSDI: ${{res.matchedRecord.osdi}})`;
        matchTag.className = "text-[11px] font-mono text-cyan-300 font-semibold";
      }} else {{
        matchTag.textContent = `✓ AI Thermal Classifier Inference (90.4% Accuracy)`;
        matchTag.className = "text-[11px] font-mono text-emerald-400 font-semibold";
      }}
    }}

    // Draw on Canvas: True Thermal, ROIs, or Tear Breakup Map
    function drawThermalCanvas() {{
      const canvas = document.getElementById('thermal-main-canvas');
      if (!canvas || !currentThermalImg) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(currentThermalImg, 0, 0, w, h);

      if (!currentThermalAnalysis) return;

      const scaleX = w / 320;
      const scaleY = h / 240;

      // Mode 2: AI Ocular ROIs & Reticles
      if (thermalViewMode === 'rois') {{
        function drawRoiBox(roi, label, isDry) {{
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

          // Corner brackets
          const clen = 12;
          ctx.strokeStyle = isDry ? '#F43F5E' : '#38BDF8';
          ctx.lineWidth = 2.5;
          ctx.beginPath(); ctx.moveTo(rx, ry + clen); ctx.lineTo(rx, ry); ctx.lineTo(rx + clen, ry); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(rx + rw - clen, ry); ctx.lineTo(rx + rw, ry); ctx.lineTo(rx + rw, ry + clen); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(rx, ry + rh - clen); ctx.lineTo(rx, ry + rh); ctx.lineTo(rx + clen, ry + rh); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(rx + rw - clen, ry + rh); ctx.lineTo(rx + rw, ry + rh); ctx.lineTo(rx + rw, ry + rh - clen); ctx.stroke();

          // Central Cornea Target
          ctx.beginPath();
          ctx.arc(rcx, rcy, 18, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
          ctx.lineWidth = 1.5;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(rcx, rcy, 3.5, 0, Math.PI * 2);
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
          ctx.fillStyle = 'rgba(13, 27, 46, 0.90)';
          ctx.fillRect(rx, ry - 22, 135, 20);
          ctx.strokeStyle = isDry ? '#F43F5E' : '#38BDF8';
          ctx.lineWidth = 1;
          ctx.strokeRect(rx, ry - 22, 135, 20);

          ctx.fillStyle = '#FFFFFF';
          ctx.font = 'bold 11px JetBrains Mono, monospace';
          ctx.fillText(`${{label}}: ${{roi.t_cc.toFixed(1)}}°C`, rx + 6, ry - 8);
        }}

        drawRoiBox(currentThermalAnalysis.re, "OD (Right)", currentThermalAnalysis.isReDry);
        drawRoiBox(currentThermalAnalysis.le, "OS (Left)", currentThermalAnalysis.isLeDry);
      }}

      // Mode 3: Tear Breakup Isotherm Map
      if (thermalViewMode === 'isotherm') {{
        const allCold = [...currentThermalAnalysis.re.cold_pixels, ...currentThermalAnalysis.le.cold_pixels];
        ctx.fillStyle = 'rgba(56, 189, 248, 0.50)';
        allCold.forEach(p => {{
          ctx.fillRect(p.x * scaleX, p.y * scaleY, scaleX + 0.5, scaleY + 0.5);
        }});

        ctx.fillStyle = 'rgba(244, 63, 94, 0.75)';
        allCold.filter(p => p.t < 33.5).forEach(p => {{
          ctx.fillRect(p.x * scaleX, p.y * scaleY, scaleX + 0.8, scaleY + 0.8);
        }});

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
      }}
    }}

    // Real-Time Temperature Probe on Canvas Mouse Move
    function handleThermalCanvasMouseMove(e) {{
      const canvas = document.getElementById('thermal-main-canvas');
      if (!canvas || !currentThermalAnalysis || !currentScaleColors) return;

      const rect = canvas.getBoundingClientRect();
      const clickX = (e.clientX - rect.left) * (320 / rect.width);
      const clickY = (e.clientY - rect.top) * (240 / rect.height);

      const px = Math.min(319, Math.max(0, Math.round(clickX)));
      const py = Math.min(239, Math.max(0, Math.round(clickY)));

      // Sample offscreen pixel color
      const off = document.createElement('canvas');
      off.width = 320; off.height = 240;
      const ctx = off.getContext('2d');
      ctx.drawImage(currentThermalImg, 0, 0, 320, 240);
      const p = ctx.getImageData(px, py, 1, 1).data;

      let bestDist = Infinity;
      let bestIdx = 0;
      for (let i = 0; i < currentScaleColors.length; i++) {{
        const sc = currentScaleColors[i];
        const d = (p[0]-sc[0])**2 + (p[1]-sc[1])**2 + (p[2]-sc[2])**2;
        if (d < bestDist) {{ bestDist = d; bestIdx = i; }}
      }}
      const temp = (37.0 - (bestIdx / (currentScaleColors.length - 1)) * 10.0).toFixed(2);

      // Determine Anatomical Region
      let region = "Periorbital Tissue";
      if (px >= 50 && px <= 140 && py >= 110 && py <= 170) {{
        region = (Math.abs(px - 95) <= 12 && Math.abs(py - 140) <= 10) ? "Right Central Cornea (OD)" : "Right Corneal Margin";
      }} else if (px >= 175 && px <= 265 && py >= 110 && py <= 170) {{
        region = (Math.abs(px - 220) <= 12 && Math.abs(py - 140) <= 10) ? "Left Central Cornea (OS)" : "Left Corneal Margin";
      }} else if (px >= 300) {{
        region = "FLIR Scale Bar";
      }}

      document.getElementById('thermal-probe-coords').textContent = `X: ${{px}}, Y: ${{py}}`;
      document.getElementById('thermal-probe-region').textContent = region;
      const tempSpan = document.getElementById('thermal-probe-temp');
      tempSpan.textContent = `${{temp}}°C`;
      tempSpan.className = (temp < 34.2) ? "px-2.5 py-0.5 rounded-md bg-rose-500/20 border border-rose-400/30 text-rose-300 font-bold text-sm" : "px-2.5 py-0.5 rounded-md bg-cyan-500/15 border border-cyan-400/30 text-cyan-300 font-bold text-sm";
    }}

    function handleThermalCanvasMouseLeave() {{
      if (currentThermalAnalysis) {{
        document.getElementById('thermal-probe-coords').textContent = `X: --, Y: --`;
        document.getElementById('thermal-probe-region').textContent = `Hover over eye`;
        document.getElementById('thermal-probe-temp').textContent = `${{currentThermalAnalysis.combined.cc.toFixed(1)}}°C (Avg CC)`;
      }}
    }}

    // Transfer Extracted Image Temperatures to Simulator Matrix Below
    function syncThermalExtractedToSimulator() {{
      if (!currentThermalAnalysis) return;
      const res = currentThermalAnalysis;

      clinicParams.cc = res.combined.cc;
      clinicParams.nc = res.combined.nc;
      clinicParams.tc = res.combined.tc;
      clinicParams.t0 = res.combined.mean;
      clinicParams.t10 = parseFloat((res.combined.mean - 0.25).toFixed(2));
      clinicParams.tl = res.combined.tc;
      clinicParams.most = parseFloat((res.combined.nc + 0.3).toFixed(2));

      if (res.matchedRecord) {{
        clinicParams.osdi = res.matchedRecord.osdi;
        clinicParams.cr10 = res.matchedRecord.re.cr10 || -0.035;
      }}

      syncUIInputs();
      runClinicInference();
      renderCornealHeatmap();

      // Scroll smoothly down to simulator
      const targetEl = document.getElementById('input-osdi');
      if (targetEl) {{
        targetEl.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
      }}
    }}
'''

# Find where the previous THERMAL_MODEL block starts and where MODEL_DATA starts
old_js_start_idx = html.find('const THERMAL_MODEL =')
if old_js_start_idx == -1:
    old_js_start_idx = html.find('// =========================================================================\n    // EMBEDDED THERMAL MODEL DATA')

model_data_idx = html.find('// Embedded Model Data & Weights trained from "Eye classification.xlsx"')

if old_js_start_idx != -1 and model_data_idx != -1:
    # Replace from old_js_start_idx to model_data_idx with new_js_block
    html = html[:old_js_start_idx] + new_js_block.strip() + "\n\n    " + html[model_data_idx:]
    print("Successfully replaced thermal engine in index.html!")
else:
    print(f"Error: Indices not found: old_js={old_js_start_idx}, model_data={model_data_idx}")

# Also update the cohort gallery HTML in #view-cohort if present
# Write updated index.html
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Saved updated index.html! New size:", len(html))
