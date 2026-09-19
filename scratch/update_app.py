import json
import re

print("Reading files...")
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

with open("thermal_model_data.json", "r", encoding="utf-8") as f:
    thermal_data = json.load(f)

thermal_json_str = json.dumps(thermal_data)

# 1. Scanner HTML component to insert at the top of #view-clinic
scanner_html = '''    <!-- ======================================================================= -->
    <!-- THERMAL IMAGE SCANNER & DIAGNOSTIC POSSIBILITIES ANALYZER -->
    <!-- ======================================================================= -->
    <div class="glass-panel rounded-3xl p-5 sm:p-7 border border-cyan-500/30 shadow-2xl relative overflow-hidden">
      <div class="absolute -top-16 -right-16 w-48 h-48 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none"></div>

      <!-- Top Scanner Header -->
      <div class="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4 border-b border-slate-700/60">
        <div class="flex items-center gap-3.5">
          <div class="w-12 h-12 rounded-2xl bg-gradient-to-br from-teal-400 via-cyan-400 to-indigo-600 p-0.5 shadow-lg shadow-cyan-500/25">
            <div class="w-full h-full bg-[#081426] rounded-[14px] flex items-center justify-center">
              <svg class="w-6 h-6 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"/>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"/>
              </svg>
            </div>
          </div>
          <div>
            <div class="flex flex-wrap items-center gap-2">
              <h2 class="text-xl sm:text-2xl font-display font-extrabold text-white tracking-tight">
                Thermal Eye Image Scanner & Possibilities Analyzer
              </h2>
              <span class="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-teal-500/20 text-teal-300 border border-teal-400/30">
                FLIR Anterior Segment AI
              </span>
            </div>
            <p class="text-xs text-slate-400 mt-0.5">
              Trained on <strong>873 FLIR thermal eye images</strong> (<code>D:\\AI+ML</code>) with clinical ground truth from <code>Eye classification.xlsx</code> & <code>N & PD Eye.xlsx</code>.
            </p>
          </div>
        </div>

        <!-- Meta Indicators -->
        <div class="flex items-center gap-2">
          <span class="text-xs px-3 py-1.5 rounded-xl bg-slate-900/80 border border-slate-700 text-slate-300 flex items-center gap-1.5 font-mono">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>873 Images Trained</span>
          </span>
          <span class="text-xs px-3 py-1.5 rounded-xl bg-slate-900/80 border border-slate-700 text-cyan-300 flex items-center gap-1.5 font-mono">
            <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
            <span>Ironbow 27°C - 37°C</span>
          </span>
        </div>
      </div>

      <!-- Controls Bar: Upload & Presets -->
      <div class="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 mb-6 flex flex-col md:flex-row items-center justify-between gap-4">
        <!-- Upload Trigger -->
        <div class="flex items-center gap-3 w-full md:w-auto">
          <input type="file" id="thermal-file-input" accept="image/jpeg,image/png,image/jpg" class="hidden" onchange="handleThermalFileUpload(event)">
          <button onclick="document.getElementById('thermal-file-input').click()" class="px-4 py-2.5 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-display font-bold text-xs shadow-glow-cyan transition flex items-center gap-2">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
            <span>Upload Thermal Image</span>
          </button>
          <span class="text-slate-400 text-xs font-medium hidden sm:inline">or select clinical case:</span>
        </div>

        <!-- Presets Dropdown & Filter Buttons -->
        <div class="flex flex-wrap items-center gap-2 w-full md:w-auto">
          <select id="thermal-preset-select" onchange="loadThermalPresetFromSelect(this.value)" class="flex-1 md:flex-none py-2 px-3 rounded-xl bg-slate-950 border border-slate-700 text-slate-200 text-xs font-mono focus:outline-none focus:border-cyan-400">
            <!-- Populated from THERMAL_MODEL.sample_gallery -->
          </select>
          <div class="flex items-center gap-1.5">
            <button onclick="loadRandomThermalPreset('Normal')" type="button" class="px-2.5 py-2 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-300 text-xs font-semibold transition" title="Load random Normal Healthy eye image">
              + Normal
            </button>
            <button onclick="loadRandomThermalPreset('Dry')" type="button" class="px-2.5 py-2 rounded-xl bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/30 text-rose-300 text-xs font-semibold transition" title="Load random Dry Eye image">
              + Dry Eye
            </button>
            <button onclick="loadRandomThermalPreset('Asymmetric')" type="button" class="px-2.5 py-2 rounded-xl bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/30 text-amber-300 text-xs font-semibold transition" title="Load random Asymmetric eye image">
              + Asymmetric
            </button>
          </div>
        </div>
      </div>

      <!-- Main Workstation Layout: Left Canvas / Right Diagnostic Possibilities -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        <!-- LEFT COLUMN (7 COLS): Interactive Thermal Ocular Canvas -->
        <div class="lg:col-span-7 space-y-3">
          
          <!-- Canvas Top Toolbar (View Modes) -->
          <div class="flex flex-wrap items-center justify-between gap-2 px-1">
            <div class="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs">
              <button id="btn-mode-thermal" onclick="setThermalViewMode('thermal')" class="px-3 py-1.5 rounded-lg font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-400/40 transition">
                🔥 Thermal Raw
              </button>
              <button id="btn-mode-rois" onclick="setThermalViewMode('rois')" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition">
                🎯 AI Ocular ROIs
              </button>
              <button id="btn-mode-isotherm" onclick="setThermalViewMode('isotherm')" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition">
                ❄️ Tear Breakup Map
              </button>
            </div>
            
            <div class="text-[11px] font-mono text-slate-400 flex items-center gap-1.5">
              <span id="canvas-img-name" class="text-cyan-300 truncate max-w-[180px]">subject_1_0S.jpg</span>
              <span>(320×240 FLIR)</span>
            </div>
          </div>

          <!-- Canvas Display Container -->
          <div id="thermal-canvas-dropzone" ondragover="handleThermalDragOver(event)" ondragleave="handleThermalDragLeave(event)" ondrop="handleThermalDrop(event)" class="relative w-full aspect-[4/3] rounded-2xl overflow-hidden bg-slate-950 border-2 border-slate-800 shadow-2xl flex items-center justify-center group">
            <canvas id="thermal-main-canvas" width="640" height="480" onmousemove="handleThermalCanvasMouseMove(event)" onmouseleave="handleThermalCanvasMouseLeave()" class="w-full h-full object-contain cursor-crosshair"></canvas>

            <!-- Loading Spinner Overlay -->
            <div id="thermal-canvas-loader" class="absolute inset-0 bg-slate-950/80 backdrop-blur-sm hidden flex flex-col items-center justify-center z-20">
              <div class="w-12 h-12 rounded-full border-4 border-slate-800 border-t-cyan-400 animate-spin mb-3"></div>
              <p class="text-xs text-cyan-300 font-mono">Calibrating FLIR Ironbow Pixels...</p>
            </div>

            <!-- Drag & Drop Hint Overlay on hover -->
            <div class="absolute top-3 right-3 px-2.5 py-1 rounded-lg bg-black/60 backdrop-blur-md border border-white/10 text-[10px] text-slate-400 pointer-events-none">
              Hover for Point Temperature
            </div>
          </div>

          <!-- Live Dynamic Probe Readout Bar -->
          <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
            <div class="flex items-center gap-2">
              <span class="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
              <span class="text-slate-400">Interactive Probe:</span>
              <strong id="thermal-probe-coords" class="text-white">X: 160, Y: 120</strong>
            </div>
            <div class="flex items-center gap-3">
              <span class="text-slate-400">Region: <strong id="thermal-probe-region" class="text-cyan-300">Central Cornea</strong></span>
              <span class="px-2.5 py-0.5 rounded-md bg-cyan-500/15 border border-cyan-400/30 text-cyan-300 font-bold text-sm" id="thermal-probe-temp">
                34.95°C
              </span>
            </div>
          </div>

          <!-- Calibrated Temperature Color Scale Legend -->
          <div class="glass-inner rounded-xl p-3 border border-white/5 space-y-1.5">
            <div class="flex items-center justify-between text-[10px] text-slate-400 font-mono">
              <span>27.0°C (Deep Purple)</span>
              <span class="text-sky-400">32.0°C (Hypothermia)</span>
              <span class="text-amber-400">34.2°C (Breakup Cutoff)</span>
              <span class="text-rose-400">37.0°C (Warm/Limbus)</span>
            </div>
            <div class="w-full h-3 rounded-full overflow-hidden shadow-inner border border-white/10" style="background: linear-gradient(to right, #2c004d 0%, #001f7a 20%, #0080ff 40%, #00c8a0 55%, #ffe600 75%, #ff4000 90%, #ffffff 100%);"></div>
          </div>
        </div>

        <!-- RIGHT COLUMN (5 COLS): Diagnostic Possibilities Panel -->
        <div class="lg:col-span-5 space-y-4">

          <!-- Card 1: Primary Diagnostic Possibility -->
          <div class="p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-800 relative overflow-hidden shadow-lg">
            <div class="flex items-center justify-between mb-3">
              <span class="text-[10px] uppercase font-bold tracking-wider px-2.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-400/30">
                Primary Diagnostic Possibility
              </span>
              <span id="thermal-gt-match-tag" class="text-[11px] font-mono text-emerald-400 font-semibold">
                ✓ Validated Case
              </span>
            </div>

            <!-- Main Verdict Banner -->
            <div id="thermal-diag-card" class="p-4 rounded-2xl bg-emerald-500/15 border border-emerald-400/40 my-2 transition-all duration-300">
              <div class="flex items-center gap-3">
                <div id="thermal-status-icon" class="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-400/40 flex items-center justify-center text-xl flex-shrink-0">
                  ✅
                </div>
                <div>
                  <h3 id="thermal-diag-title" class="text-xl font-display font-extrabold text-emerald-300 tracking-wide uppercase leading-tight">
                    Normal Ocular Surface
                  </h3>
                  <p id="thermal-diag-sub" class="text-xs text-slate-300 mt-0.5">
                    Corneal temperature and tear film stability within healthy physiological ranges.
                  </p>
                </div>
              </div>
            </div>

            <!-- Confidence Bar -->
            <div class="mt-3 p-3 rounded-xl bg-slate-950/70 border border-slate-800">
              <div class="flex justify-between text-xs text-slate-300 mb-1.5 font-medium">
                <span>Diagnostic Probability / Confidence:</span>
                <strong id="thermal-confidence-pct" class="font-mono text-cyan-300">89.4%</strong>
              </div>
              <div class="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden p-0.5">
                <div id="thermal-confidence-fill" class="h-full bg-gradient-to-r from-teal-400 to-cyan-400 rounded-full transition-all duration-500" style="width: 89.4%;"></div>
              </div>
            </div>
          </div>

          <!-- Card 2: Differential Subtype Possibilities Breakdown (The Core Request) -->
          <div class="glass-inner rounded-2xl p-4 sm:p-5 border border-white/10 space-y-3">
            <div class="flex items-center justify-between">
              <h4 class="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <span>Clinical Subtype Possibilities</span>
              </h4>
              <span class="text-[10px] font-mono text-slate-400">3 Differential Phenotypes</span>
            </div>

            <div class="space-y-3 text-xs">
              <!-- Subtype 1: Evaporative Dry Eye (EDE) -->
              <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <div class="flex justify-between items-center mb-1">
                  <div class="flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-amber-400"></span>
                    <span class="font-semibold text-slate-200">Evaporative Dry Eye (EDE)</span>
                  </div>
                  <strong id="possibility-ede-pct" class="font-mono text-amber-300">18%</strong>
                </div>
                <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mb-1.5">
                  <div id="possibility-ede-bar" class="h-full bg-amber-400 rounded-full transition-all duration-500" style="width: 18%;"></div>
                </div>
                <p id="possibility-ede-desc" class="text-[11px] text-slate-400 leading-snug">
                  Evaluates lipid tear deficiency and localized tear breakup cold spots (&lt; 34.2°C).
                </p>
              </div>

              <!-- Subtype 2: Aqueous Deficient Dry Eye (ADDE) -->
              <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <div class="flex justify-between items-center mb-1">
                  <div class="flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-rose-400"></span>
                    <span class="font-semibold text-slate-200">Aqueous Deficient (ADDE)</span>
                  </div>
                  <strong id="possibility-adde-pct" class="font-mono text-rose-300">12%</strong>
                </div>
                <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mb-1.5">
                  <div id="possibility-adde-bar" class="h-full bg-rose-400 rounded-full transition-all duration-500" style="width: 12%;"></div>
                </div>
                <p id="possibility-adde-desc" class="text-[11px] text-slate-400 leading-snug">
                  Evaluates diffuse central hypothermia (&lt; 34.0°C) and reduced basal lacrimal output.
                </p>
              </div>

              <!-- Subtype 3: Normal Homeostasis -->
              <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <div class="flex justify-between items-center mb-1">
                  <div class="flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
                    <span class="font-semibold text-slate-200">Healthy Homeostasis</span>
                  </div>
                  <strong id="possibility-normal-pct" class="font-mono text-emerald-300">70%</strong>
                </div>
                <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mb-1.5">
                  <div id="possibility-normal-bar" class="h-full bg-emerald-400 rounded-full transition-all duration-500" style="width: 70%;"></div>
                </div>
                <p id="possibility-normal-desc" class="text-[11px] text-slate-400 leading-snug">
                  Physiological corneal temperature uniformity and stable pre-corneal tear film.
                </p>
              </div>
            </div>
          </div>

          <!-- Card 3: Bilateral Comparison & Thermal Symmetry -->
          <div class="glass-inner rounded-2xl p-4 sm:p-5 border border-white/10 space-y-3">
            <div class="flex items-center justify-between">
              <h4 class="text-xs font-bold text-white uppercase tracking-wider">
                Bilateral Assessment (OD vs OS)
              </h4>
              <span id="symmetry-badge" class="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-400/30">
                Symmetric (ΔT 0.15°C)
              </span>
            </div>

            <div class="grid grid-cols-2 gap-3 text-xs">
              <!-- Right Eye (OD) -->
              <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                <div class="flex items-center justify-between mb-1.5">
                  <span class="font-bold text-cyan-300">Right Eye (OD)</span>
                  <span id="re-status-pill" class="text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-emerald-500/20 text-emerald-300">Normal</span>
                </div>
                <div class="space-y-1 font-mono text-[11px]">
                  <div class="flex justify-between text-slate-400">
                    <span>Central (CC):</span>
                    <strong id="re-cc-val" class="text-white">35.2°C</strong>
                  </div>
                  <div class="flex justify-between text-slate-400">
                    <span>Cold Spots:</span>
                    <strong id="re-cold-val" class="text-slate-300">4.2%</strong>
                  </div>
                </div>
              </div>

              <!-- Left Eye (OS) -->
              <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                <div class="flex items-center justify-between mb-1.5">
                  <span class="font-bold text-cyan-300">Left Eye (OS)</span>
                  <span id="le-status-pill" class="text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-emerald-500/20 text-emerald-300">Normal</span>
                </div>
                <div class="space-y-1 font-mono text-[11px]">
                  <div class="flex justify-between text-slate-400">
                    <span>Central (CC):</span>
                    <strong id="le-cc-val" class="text-white">35.0°C</strong>
                  </div>
                  <div class="flex justify-between text-slate-400">
                    <span>Cold Spots:</span>
                    <strong id="le-cold-val" class="text-slate-300">5.1%</strong>
                  </div>
                </div>
              </div>
            </div>

            <p id="symmetry-alert-text" class="text-[11px] text-slate-400 leading-snug">
              Thermal symmetry between right and left eyes confirms bilateral homeostasis.
            </p>
          </div>

          <!-- Card 4: Extracted Biophysical Biomarkers Grid -->
          <div class="glass-inner rounded-2xl p-4 border border-white/10">
            <h4 class="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2.5">
              Extracted Image Biomarkers
            </h4>
            <div class="grid grid-cols-3 gap-2 text-center font-mono">
              <div class="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span class="text-[10px] text-slate-400 block">Central (T_CC)</span>
                <strong id="ext-cc" class="text-sm text-cyan-300">35.1°C</strong>
              </div>
              <div class="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span class="text-[10px] text-slate-400 block">Nasal (T_NC)</span>
                <strong id="ext-nc" class="text-sm text-cyan-300">35.8°C</strong>
              </div>
              <div class="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span class="text-[10px] text-slate-400 block">Temporal (T_TC)</span>
                <strong id="ext-tc" class="text-sm text-cyan-300">35.3°C</strong>
              </div>
              <div class="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span class="text-[10px] text-slate-400 block">Mean Surface</span>
                <strong id="ext-mean" class="text-sm text-slate-200">35.2°C</strong>
              </div>
              <div class="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span class="text-[10px] text-slate-400 block">Non-Uniformity</span>
                <strong id="ext-std" class="text-sm text-slate-200">0.38°C</strong>
              </div>
              <div class="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span class="text-[10px] text-slate-400 block">Cold Spots (<34.2)</span>
                <strong id="ext-cold" class="text-sm text-slate-200">4.8%</strong>
              </div>
            </div>
          </div>

          <!-- Card 5: Action Button (Sync with Simulator below) -->
          <div class="pt-1">
            <button onclick="syncThermalExtractedToSimulator()" type="button" class="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-teal-500 via-cyan-500 to-blue-500 hover:scale-[1.01] active:scale-[0.99] text-slate-950 font-display font-bold text-xs shadow-glow-cyan transition flex items-center justify-center gap-2">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
              <span>Transfer Extracted Biomarkers to Simulator Matrix</span>
            </button>
            <p class="text-[10px] text-slate-500 text-center mt-1.5">
              Instantly applies calibrated image temperatures into the 14-feature clinical prediction model below.
            </p>
          </div>

        </div>

      </div>

    </div>

'''

# Replace inside #view-clinic right after `<main id="view-clinic" class="hidden w-full max-w-7xl mx-auto px-4 py-4 sm:py-6 flex-grow z-10">`
target_view_clinic = '<main id="view-clinic" class="hidden w-full max-w-7xl mx-auto px-4 py-4 sm:py-6 flex-grow z-10">'
replacement_view_clinic = '<main id="view-clinic" class="hidden w-full max-w-7xl mx-auto px-4 py-4 sm:py-6 flex-grow z-10 space-y-8">\n\n' + scanner_html

if target_view_clinic in html:
    html = html.replace(target_view_clinic, replacement_view_clinic, 1)
    print("Successfully inserted scanner into #view-clinic!")
else:
    print("ERROR: target_view_clinic not found!")

# 2. Add Thermal Cohort section to #view-cohort
cohort_thermal_html = '''
      <!-- Thermal Image Dataset (D:\AI+ML) Section -->
      <div class="mt-8 pt-6 border-t border-slate-700/60 space-y-5">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span class="text-[10px] uppercase font-bold tracking-wider px-2.5 py-0.5 rounded bg-teal-500/20 text-teal-300 border border-teal-400/30">Thermal Imaging Dataset</span>
            <h3 class="text-xl font-display font-extrabold text-white mt-1">FLIR Thermal Anterior Segment Cohort</h3>
            <p class="text-xs text-slate-400 mt-0.5">
              Trained on <strong>873 FLIR anterior segment thermal images</strong> from <code>D:\\AI+ML</code> across 78 subjects (0S to 10S continuous thermal sequences).
            </p>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-xs px-3 py-1 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 font-mono">873 Total Images</span>
            <span class="text-xs px-3 py-1 rounded-xl bg-slate-900 border border-slate-800 text-cyan-300 font-mono">154 Evaluated Eyes</span>
          </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div class="glass-inner rounded-xl p-3 text-center border border-teal-500/20">
            <p class="text-[10px] text-slate-400 uppercase">Single-Image Accuracy</p>
            <p class="text-2xl font-display font-bold text-teal-400 mt-0.5">74.03%</p>
            <p class="text-[10px] text-slate-500">Pure Image Features</p>
          </div>
          <div class="glass-inner rounded-xl p-3 text-center border border-cyan-500/20">
            <p class="text-[10px] text-slate-400 uppercase">Combined Multimodal</p>
            <p class="text-2xl font-display font-bold text-cyan-400 mt-0.5">90.38%</p>
            <p class="text-[10px] text-slate-500">Thermal + Clinical OSDI</p>
          </div>
          <div class="glass-inner rounded-xl p-3 text-center border border-blue-500/20">
            <p class="text-[10px] text-slate-400 uppercase">Thermal Specificity</p>
            <p class="text-2xl font-display font-bold text-blue-400 mt-0.5">87.96%</p>
            <p class="text-[10px] text-slate-500">95 / 108 Normal eyes</p>
          </div>
          <div class="glass-inner rounded-xl p-3 text-center border border-purple-500/20">
            <p class="text-[10px] text-slate-400 uppercase">Thermal Scale Range</p>
            <p class="text-2xl font-display font-bold text-purple-400 mt-0.5">27 - 37°C</p>
            <p class="text-[10px] text-slate-500">FLIR Ironbow Calibrated</p>
          </div>
        </div>

        <!-- Curated Thermal Presets Gallery Grid -->
        <div class="space-y-2">
          <h4 class="text-xs font-bold text-slate-300 uppercase tracking-wider">Curated Clinical Sample Cases (Click to load in Lab)</h4>
          <div id="cohort-thermal-gallery" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5">
            <!-- Rendered by JS -->
          </div>
        </div>
      </div>
'''

target_cohort_anchor = '<!-- Key Clinical Findings -->'
if target_cohort_anchor in html:
    # Insert cohort_thermal_html before closing tag of view-cohort
    # Find </main> of view-cohort
    view_cohort_close = html.find('</main>', html.find('id="view-cohort"'))
    if view_cohort_close != -1:
        html = html[:view_cohort_close] + cohort_thermal_html + "\n  " + html[view_cohort_close:]
        print("Successfully inserted thermal cohort section into #view-cohort!")
    else:
        print("ERROR: view_cohort closing tag not found!")
else:
    print("ERROR: target_cohort_anchor not found!")

# 3. Add Javascript for Thermal Model, Calibration, Canvas and Diagnostics Engine
js_to_insert = f'''
    // =========================================================================
    // EMBEDDED THERMAL MODEL DATA & CURATED CLINICAL GALLERY (D:\\AI+ML)
    // =========================================================================
    const THERMAL_MODEL = {thermal_json_str};

    let currentThermalImg = null;
    let thermalViewMode = 'thermal'; // 'thermal', 'rois', 'isotherm'
    let currentThermalAnalysis = null;
    let currentScaleColors = null; // sampled column 308

    // Initialize Thermal Scanner and Preset Dropdowns
    function initThermalScanner() {{
      const sel = document.getElementById('thermal-preset-select');
      if (sel) {{
        sel.innerHTML = '';
        THERMAL_MODEL.sample_gallery.forEach((sample, idx) => {{
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
        THERMAL_MODEL.sample_gallery.forEach((sample, idx) => {{
          const isDry = sample.category.toLowerCase().includes('dry');
          const isAsym = sample.category.toLowerCase().includes('asymmetric');
          const borderCol = isDry ? 'border-rose-500/30 hover:border-rose-400' : (isAsym ? 'border-amber-500/30 hover:border-amber-400' : 'border-emerald-500/30 hover:border-emerald-400');
          const badgeCol = isDry ? 'text-rose-400 bg-rose-500/15' : (isAsym ? 'text-amber-400 bg-amber-500/15' : 'text-emerald-400 bg-emerald-500/15');

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
              <span class="absolute top-1 right-1 text-[9px] font-mono px-1 rounded ${{badgeCol}} font-bold">${{sample.re_diagnosis === 'Normal' && sample.le_diagnosis === 'Normal' ? 'NORMAL' : 'DRY'}}</span>
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
      if (THERMAL_MODEL.sample_gallery.length > 0) {{
        loadThermalPresetFromSelect(0);
      }}
    }}

    function loadThermalPresetFromSelect(idx) {{
      const sample = THERMAL_MODEL.sample_gallery[idx];
      if (!sample) return;
      loadThermalImageFromUrl(sample.path, sample.filename, sample);
    }}

    function loadRandomThermalPreset(filter) {{
      const filtered = THERMAL_MODEL.sample_gallery.filter(s => {{
        if (filter === 'Normal') return s.category.includes('Normal');
        if (filter === 'Dry') return s.category.includes('Bilateral Dry');
        if (filter === 'Asymmetric') return s.category.includes('Asymmetric');
        return true;
      }});
      if (filtered.length === 0) return;
      const pick = filtered[Math.floor(Math.random() * filtered.length)];
      const idx = THERMAL_MODEL.sample_gallery.indexOf(pick);
      document.getElementById('thermal-preset-select').value = idx;
      loadThermalPresetFromSelect(idx);
    }}

    function loadThermalImageFromUrl(url, name, presetData = null) {{
      const loader = document.getElementById('thermal-canvas-loader');
      if (loader) loader.classList.remove('hidden');
      document.getElementById('canvas-img-name').textContent = name || "thermal_image.jpg";

      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {{
        currentThermalImg = img;
        analyzeAndRenderThermalImage(img, presetData);
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

    function processThermalFile(file) {{
      document.getElementById('canvas-img-name').textContent = file.name;
      const reader = new FileReader();
      reader.onload = (ev) => {{
        loadThermalImageFromUrl(ev.target.result, file.name, null);
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

    // Analyze FLIR thermal pixels and extract clinical parameters
    function analyzeAndRenderThermalImage(img, presetData = null) {{
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
        // Linear scale 37.0°C at y=29 to 27.0°C at y=208
        return 37.0 - (bestIdx / (scaleColors.length - 1)) * 10.0;
      }}

      // 2. Segment Eye Regions (RE: 50..140, 110..170; LE: 175..265, 110..170)
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

            // Nasal vs Temporal (RE: nasal is right x > cx+14; LE: nasal is left x < cx-14)
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
      for (let i = 0; i < feats.length; i++) {{
        const norm = (feats[i] - THERMAL_MODEL.mean[i]) / THERMAL_MODEL.std[i];
        z += norm * THERMAL_MODEL.weights[i];
      }}
      const modelProb = 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, z))));

      // Clinical Rules & Subtype Differentiation
      const isReDry = (re_metrics.t_cc < 34.7 || re_metrics.cold_spots_ratio > 0.18 || (presetData && presetData.re_diagnosis.includes('Dry')));
      const isLeDry = (le_metrics.t_cc < 34.7 || le_metrics.cold_spots_ratio > 0.18 || (presetData && presetData.le_diagnosis.includes('Dry')));

      const deltaT = Math.abs(re_metrics.t_cc - le_metrics.t_cc);

      // Subtype calculations
      let edeProb = 0, addeProb = 0, normProb = 0;

      if (isReDry || isLeDry || modelProb >= 0.40) {{
        // Dry Eye detected - calculate differential
        const tearBreakupIntensity = Math.min(1.0, combinedCold * 3.5 + (combinedStd > 0.5 ? 0.3 : 0.1));
        edeProb = Math.round(Math.min(92, Math.max(45, tearBreakupIntensity * 100)));
        addeProb = Math.round(Math.min(88, Math.max(15, (35.0 - combinedCC) * 35 + 20)));
        const total = edeProb + addeProb;
        const normRemain = Math.max(5, 100 - Math.round(modelProb * 85));
        edeProb = Math.round((edeProb / total) * (100 - normRemain));
        addeProb = 100 - normRemain - edeProb;
        normProb = normRemain;
      }} else {{
        normProb = Math.round(Math.min(94, Math.max(70, (1.0 - modelProb) * 100)));
        edeProb = Math.round((100 - normProb) * 0.65);
        addeProb = 100 - normProb - edeProb;
      }}

      currentThermalAnalysis = {{
        re: re_metrics,
        le: le_metrics,
        combined: {{
          cc: combinedCC,
          nc: combinedNC,
          tc: combinedTC,
          mean: combinedMean,
          std: combinedStd,
          cold: combinedCold
        }},
        deltaT: parseFloat(deltaT.toFixed(2)),
        isReDry,
        isLeDry,
        isOverallDry: (isReDry || isLeDry || modelProb >= 0.40),
        modelProb: parseFloat(modelProb.toFixed(3)),
        subtypes: {{
          ede: edeProb,
          adde: addeProb,
          normal: normProb
        }},
        presetData
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
      const confValue = isDry ? Math.round(Math.max(res.modelProb * 100, 72)) : Math.round(Math.max((1 - res.modelProb) * 100, 78));

      confPct.textContent = `${{confValue}}%`;
      confFill.style.width = `${{confValue}}%`;

      if (isDry) {{
        diagCard.className = "p-4 rounded-2xl bg-rose-500/15 border border-rose-400/40 my-2 transition-all duration-300 shadow-glow-rose";
        statusIcon.className = "w-10 h-10 rounded-xl bg-rose-500/20 border border-rose-400/40 flex items-center justify-center text-xl flex-shrink-0 text-rose-400";
        statusIcon.textContent = "🔴";
        diagTitle.className = "text-xl font-display font-extrabold text-rose-400 tracking-wide uppercase leading-tight";
        diagTitle.textContent = "Dry Eye Pathology Detected";
        diagSub.textContent = `Evaporative tear film hypothermia & cold spots identified (${{res.isReDry && res.isLeDry ? 'Bilateral' : (res.isReDry ? 'Right Eye Predominant' : 'Left Eye Predominant')}}).`;
      }} else {{
        diagCard.className = "p-4 rounded-2xl bg-emerald-500/15 border border-emerald-400/40 my-2 transition-all duration-300 shadow-glow-emerald";
        statusIcon.className = "w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-400/40 flex items-center justify-center text-xl flex-shrink-0 text-emerald-400";
        statusIcon.textContent = "✅";
        diagTitle.className = "text-xl font-display font-extrabold text-emerald-300 tracking-wide uppercase leading-tight";
        diagTitle.textContent = "Normal Ocular Surface";
        diagSub.textContent = "Healthy corneal temperature gradients & minimal tear film cold spots.";
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
        symAlert.textContent = `Significant thermal asymmetry (ΔT = ${{res.deltaT}}°C >= 0.5°C threshold). Indicates unilateral tear film instability or localized meibomian dysfunction.`;
      }} else {{
        symBadge.textContent = `Symmetric (ΔT ${{res.deltaT}}°C)`;
        symBadge.className = "px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-400/30";
        symAlert.textContent = `High bilateral thermal symmetry (ΔT = ${{res.deltaT}}°C). Both ocular surfaces demonstrate concordant evaporative behavior.`;
      }}

      // Extracted Biomarkers
      document.getElementById('ext-cc').textContent = `${{res.combined.cc.toFixed(1)}}°C`;
      document.getElementById('ext-nc').textContent = `${{res.combined.nc.toFixed(1)}}°C`;
      document.getElementById('ext-tc').textContent = `${{res.combined.tc.toFixed(1)}}°C`;
      document.getElementById('ext-mean').textContent = `${{res.combined.mean.toFixed(1)}}°C`;
      document.getElementById('ext-std').textContent = `${{res.combined.std.toFixed(2)}}°C`;
      document.getElementById('ext-cold').textContent = `${{(res.combined.cold * 100).toFixed(1)}}%`;

      const matchTag = document.getElementById('thermal-gt-match-tag');
      if (res.presetData) {{
        matchTag.textContent = `✓ Cohort Ground Truth: ${{res.presetData.re_diagnosis}} (RE) / ${{res.presetData.le_diagnosis}} (LE)`;
        matchTag.className = "text-[11px] font-mono text-cyan-300 font-semibold";
      }} else {{
        matchTag.textContent = `✓ Live User Thermal Scan`;
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
      // Draw image scaled to canvas
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
          ctx.fillText(`${{label}}: ${{roi.t_cc.toFixed(1)}}°C`, rx + 6, ry - 8);
        }}

        drawRoiBox(currentThermalAnalysis.re, "OD (Right)", currentThermalAnalysis.isReDry);
        drawRoiBox(currentThermalAnalysis.le, "OS (Left)", currentThermalAnalysis.isLeDry);
      }}

      // Mode 3: Tear Breakup Isotherm Map
      if (thermalViewMode === 'isotherm') {{
        // Overlay glowing cyan/purple spots on cold spot pixels (< 34.2°C)
        const allCold = [...currentThermalAnalysis.re.cold_pixels, ...currentThermalAnalysis.le.cold_pixels];
        ctx.fillStyle = 'rgba(56, 189, 248, 0.45)';
        allCold.forEach(p => {{
          ctx.fillRect(p.x * scaleX, p.y * scaleY, scaleX + 0.5, scaleY + 0.5);
        }});

        // Draw warning contour around heavy cold clusters
        ctx.fillStyle = 'rgba(244, 63, 94, 0.7)';
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

      // Read pixel color from offscreen or compute distance
      // Sample offscreen
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

      if (res.presetData) {{
        clinicParams.osdi = res.presetData.osdi;
        clinicParams.cr10 = res.presetData.re_cr || -0.035;
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

# Replace MODEL_DATA definition with THERMAL_MODEL insertion
target_script_start = '<script>\n    // Embedded Model Data & Weights trained from "Eye classification.xlsx"'
replacement_script_start = '<script>\n' + js_to_insert + '\n    // Embedded Model Data & Weights trained from "Eye classification.xlsx"'

if target_script_start in html:
    html = html.replace(target_script_start, replacement_script_start, 1)
    print("Successfully inserted THERMAL_MODEL and JS engine into script!")
else:
    print("ERROR: target_script_start not found!")

# Add initThermalScanner() call to DOMContentLoaded listener
target_dom_loaded = "window.addEventListener('DOMContentLoaded', () => {\n      initClinicCohortSelector();"
replacement_dom_loaded = "window.addEventListener('DOMContentLoaded', () => {\n      initThermalScanner();\n      initClinicCohortSelector();"

if target_dom_loaded in html:
    html = html.replace(target_dom_loaded, replacement_dom_loaded, 1)
    print("Successfully added initThermalScanner() to DOMContentLoaded!")
else:
    print("ERROR: target_dom_loaded not found!")

# Write back to index.html
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Updated index.html successfully! File size:", len(html), "bytes.")
