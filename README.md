# Perth Mint Gold Forecasting Pipeline - Execution Guide

A robust, two-part deep learning framework that dynamically pulls live transactional spot entries directly from the official Perth Mint database API and utilizes a high-capacity, self-optimizing 4-layer neural network with an Adam optimizer to calculate tomorrow's gold price spot forecast.

---

## 🛠️ 1. System Requirements & Dependencies

Before running the pipeline, make sure your Python workspace environment has the standard data parsing and mathematical array computation stacks installed. Open your terminal prompt and execute:

```bash
pip install numpy pandas requests beautifulsoup4
```

---

## 📂 2. File Architecture Alignment

To prevent system lookup path errors, ensure both script files are saved inside the **exact same workspace directory folder** (e.g., `d:\AICoding\pythonCoding\p1\`):

*   `preprocessor.py` — Directly connects to the live API URL, cleans row line delimiters, and isolates the target columns.
*   `advanced_brain_engine.py` — Runs the grid cross-validation search, standardizes Z-scores, executes the 40,000-epoch training sequence, and generates the forecast.

---

## 🚀 3. How to Run the Scripts

Open your terminal or command prompt, navigate into your active development folder, and launch the master engine script file:

```bash
# Navigate to your workspace folder
cd d:\AICoding\pythonCoding\p1\

# Execute the core machine learning pipeline
python advanced_brain_engine.py
```

---

## 📊 4. What Happens Upon Execution?

Once triggered, all data processing passes run cleanly and automatically sequentially:

1. **Live Data Download**: The pipeline contacts the secure Perth Mint endpoint (`range=days`) and immediately logs the data line-by-line onto your terminal window for structural tracking.
2. **Hyperparameter Grid Search**: The model tests multiple multi-day lookback windows and layered network dimensions across an 80/20 chronological data split to find the layout with the absolute lowest validation Root Mean Squared Error (RMSE).
3. **Intensive Matrix Tuning (40,000 Epochs)**: The brain instantiates the selected optimal deep network structure and executes 40,000 training iterations over the dataset completely silently to maximize runtime processing performance.
4. **Final Target Forecast**: The network pipes the single freshest data vector through the trained matrix weights and displays your clean analytical summary card directly onto the console window.

---

## 📋 5. Expected Terminal Workspace View

```text
=== [DIAGNOSTIC DISPLAY] PRINTING ALL DATA READ FROM THE CSV ===
Row 01 | Date-Time: 06/02/2026 00:59:59 | Price 1 (Isolated): \$6289.71
Row 02 | Date-Time: 07/02/2026 00:59:59 | Price 1 (Isolated): \$6295.20
...
Row 22 | Date-Time: 09/06/2026 14:15:00 | Price 1 (Isolated): \$6145.04
===================================================================

[+] Initializing Extensive Hyperparameter Grid Search over historical arrays...
[✔ Search Concluded] Lowest Out-of-Sample Validation RMSE achieved: 3.1245
    Selected Optimal Horizon: 6-day lookback window.
    Selected Optimal Layer Architecture: Hidden [32, 16]
[+] Launching long-form deep optimization sequence (40,000 Epochs)...

==================================================
    THE PERTH MINT API REAL-TIME GOLD FORECAST    
==================================================
  Real-time Value (Today's Close): \$6145.04 AUD
  Tomorrow Price (Predicted Target):  \$6151.18 AUD
==================================================
```
