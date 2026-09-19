# Dry Eye AI — Clinical Detection & Thermography System

An intelligent clinical decision support and patient screening platform for Dry Eye Disease (DED), combining validated **7-day symptom questionnaire (with Laterality)**, **anterior segment FLIR thermal imaging AI**, and a **multimodal machine learning classifier** trained on clinical ocular thermography data and thermal image sequences.

---

## 🚀 Quick Start

### 1. View Web Application
The local HTTP server is running at:
- **URL**: [http://localhost:8000/index.html](http://localhost:8000/index.html)
- You can also open [index.html](file:///c:/Users/ezhil/dry%20eye%20finder/index.html) directly in any modern web browser.

### 2. Retrain Thermal Image Model
To re-process thermal images from `D:\AI+ML` and re-train the anterior segment classifier:
```bash
python train_thermal_model.py
```

### 3. Retrain Multimodal Clinical Classifier
To re-train the clinical biomarker classifier from `Eye classification.xlsx` and `N & PD Eye.xlsx`:
```bash
python train_and_export_model.py
```

---

## 🔬 System Components & Architecture

### 1. Patient Screening Wizard (`Patient Screening`)
- **7 Clinical Symptom Questions (Past 7 Days Recall)**:
  1. *Ocular Discomfort*: Gritty, painful, or sore eyes.
  2. *Visual Sensitivity*: Sensitivity to light.
  3. *Environmental Triggers*: Windy conditions or low humidity.
  4. *Environmental Exposure*: Air-conditioned environments.
  5. *Visual & Digital Activities*: TV, computer, or reading.
  6. *Night-Time Discomfort*: Night driving visual discomfort.
  7. *Symptom Laterality*: Right eye only, Left eye only, or Both eyes.
- **Automated OSDI Scoring**: Calculates standard 0–100 index score with severity grading (`Normal`, `Mild`, `Moderate`, `Severe`).

### 2. Clinical Thermography Diagnostic Lab (`Thermography Lab`)
- **Thermal Eye Image Scanner & Possibilities Analyzer**:
  - **Custom Image Upload**: Drag & drop or upload ANY thermal JPEG/PNG from `D:\AI+ML` or local disk.
  - **Clinical Preset Gallery**: 15 pre-loaded verified subject cases (Bilateral Dry Eye, Healthy Normal, and Asymmetric cases).
  - **Interactive High-Resolution Thermal Canvas**:
    - `🔥 Thermal Raw`: Calibrated FLIR thermal palette display.
    - `🎯 AI Ocular ROIs`: Precise corneal segmentation with Central ($T_{CC}$), Nasal ($T_{NC}$), and Temporal ($T_{TC}$) reticles and temperature badges for OD (RE) and OS (LE).
    - `❄️ Tear Breakup Isotherm Map`: Real-time cold-spot overlay highlighting tear film breakup zones ($< 34.2^\circ\text{C}$).
    - `📍 Dynamic Cursor Probe`: Real-time crosshair coordinate tracking and temperature readout on hover.
  - **Diagnostic Possibilities Panel**:
    - **Primary Diagnostic Verdict**: `Normal Ocular Surface` vs `Dry Eye Disease Detected` with confidence level percentage.
    - **Differential Subtype Possibilities (3 Phenotypes)**:
      - *Evaporative Dry Eye (EDE)*: Evaluates tear film lipid layer disruption and localized cold spots.
      - *Aqueous Deficient Dry Eye (ADDE)*: Evaluates diffuse central hypothermia ($< 34.0^\circ\text{C}$) and basal tear deficiency.
      - *Healthy Homeostasis*: Evaluates physiological thermal stability.
    - **Bilateral Assessment (OD vs OS)**: Eye-by-eye corneal temperatures, cold-spot ratios, and **Thermal Symmetry Index** ($\Delta T = |T_{OD} - T_{OS}|$ with $\ge 0.5^\circ\text{C}$ clinical asymmetry alerts).
    - **Extracted Biophysical Biomarkers**: $T_{CC}, T_{NC}, T_{TC}, T_{mean}, T_{std}$ (heterogeneity), and cold-spot area.
    - **"Transfer Extracted Biomarkers to Simulator Matrix"**: 1-click sync into the 14-feature multimodal simulator below.

### 3. Multimodal Biomarker Matrix & Real-Time Parameter Simulator
- **Interactive Parameter Sliders**: Central Cornea ($CC$), Nasal Cornea ($NC$), Temporal Cornea ($TC$), Cooling Rate at 10s ($CR_{10S}$), Sustained 10s Temp ($T_{10S}$), MOST, and OSDI.
- **Explainable AI (SHAP-Proxy)**: Real-time biomarker attribution showing which physical factors push the diagnosis toward Dry Eye or Normal.


---

## 📁 Repository Structure

```
dry eye finder/
├── index.html                  # Dry Eye AI web application with Thermal Scanner & Possibilities Analyzer
├── train_thermal_model.py      # Thermal image processing, calibration & model training pipeline
├── thermal_model_data.json     # Serialized thermal weights, calibration constants & curated samples
├── train_and_export_model.py   # Clinical biomarker tabular model training script
├── model_data.json             # Serialized multimodal clinical model data
├── inspect_excel.py            # Excel ground-truth inspector script
├── samples/                    # 15 web-accessible curated FLIR thermal sample images from D:\AI+ML
└── README.md                   # System documentation & technical reference
```
