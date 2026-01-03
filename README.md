# Symbolic Regression and Misfit-Guided Modeling of B2 Phase Stability

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper: Acta Materialia](https://img.shields.io/badge/Paper-Acta%20Materialia-red)](https://doi.org/10.1016/j.actamat.2025.XXXXXX)

**Companion repository for the research article: "Symbolic Regression and Misfit-Guided Modeling of B2 Phase Stability in Ru-Containing Refractory Alloys"**

## 📖 Overview

This repository contains the dataset, machine learning (Random Forest) models, and Symbolic Regression (PySR) workflows used to discover the physical laws governing phase stability in Ruthenium-containing refractory superalloys.

By integrating high-throughput DFT calculations with interpretable AI, we derive a closed-form expression for the B2 solvus temperature that explicitly decouples thermodynamic driving forces from elastic coherency penalties:

$$T_{\text{solvus}} \approx 0.11 \left( \frac{\Delta H}{\Delta S_{\text{mix}}} \right) - 20000 \cdot |\delta|$$

This framework resolves the "Binary Paradox" in refractory alloy design and identifies multi-component "lattice tuning" strategies to enable ultra-high-temperature stability.

---

## 📂 Repository Structure

```text
├── data/
│   ├── Final_with_Tsstar_fixed.csv      # Raw DFT training data (Enthalpy, Lattice Params)
│   ├── All_Predictions_V10.csv          # Full ML screening of ~3,500 candidate systems
│   ├── Ideal_Candidates_V10.csv         # Top-ranked zero-misfit candidates (Ru-Hf-Al, etc.)
│   └── hall_of_fame.csv                 # Output equations from PySR symbolic search
│
├── scripts/
│   ├── solvus_sr_pipeline_v10.py        # Main Symbolic Regression pipeline (using PySR)
│   ├── plot_volcano.py                  # Generates the Misfit Volcano Plot (Fig 2)
│   ├── plot_design_map.py               # Generates the Stability Heatmaps (Fig 6)
│   └── plot_elemental_trends.py         # Generates Elemental contribution plots (Fig 8)
│
├── figures/                             # Generated high-resolution plots used in the manuscript
│   ├── SR-Optimization.png
│   ├── ML-Optimization-design.png
│   └── Elemental_Volcano.png
│
├── requirements.txt                     # Python dependencies
└── README.md

Key Findings

    The Coherency Penalty: We discovered a massive linear penalty coefficient (20,000) for lattice misfit. A mere 1% misfit reduces the solvus temperature by ~200°C.

    Lattice Tuning: Stoichiometric binaries (e.g., pure RuHf) fail due to high misfit (>5%). Stability is only achieved by alloying with "lattice tuning agents" (Al, Ti) that reduce the precipitate molar volume to match the BCC matrix.

    Design Strategy: The code provides a zero-misfit design protocol: fix the matrix VEC ≈ 5.5, then tune the precipitate chemistry to minimize ∣δ∣→0.

🛠️ Installation & Usage
1. Prerequisites

This workflow requires Python 3.8+ and the following libraries:

    numpy, pandas, scipy

    scikit-learn (for Random Forest)

    matplotlib, seaborn (for plotting)

    pysr (for Symbolic Regression)

Install dependencies:
Bash

pip install -r requirements.txt

Note: PySR requires a working Julia installation in the background. See PySR documentation.
2. Random Forest Screening (Step 1)

To train the Random Forest surrogate model on the DFT data and generate predictions for the full 3,500 alloy candidates:
Bash

python scripts/train_rf_model.py --input data/Final_with_Tsstar_fixed.csv --output data/All_Predictions_V10.csv

This script performs hyperparameter tuning (GridSearchCV), evaluates feature importance (SHAP/MDI), and saves the trained model to models/rf_solvus_model.pkl.
3. Symbolic Regression Search (Step 2)

To run the "Guided Freedom" search that extracts the governing physics law from the screened data:
Bash

python scripts/solvus_sr_pipeline_v10.py --input data/All_Predictions_V10.csv --outdir sr_results

4. Reproducing the Figures

To generate the "Volcano Plot" showing the coherency window:
Bash

python scripts/plot_volcano.py

To generate the Elemental Breakdown (Ru-Al, Ru-Ti, Ru-Hf trends):
Bash

python scripts/plot_elemental_trends.py

📊 Data Description

    Misfit / lattice_misfit_delta: The coherent lattice mismatch between relaxed B2 and BCC phases.

    Term_Thermo_Ratio: The ratio of Formation Enthalpy to Mixing Entropy (ΔH/ΔS).

    Predicted_Solvus_C: The predicted solvus temperature in Celsius based on the RF/SR model.

    B2_Ru, B2_Hf, etc.: Atomic fractions of elements in the B2 phase.

📜 Citation

If you use this data or code in your research, please cite the following paper:

    Mahata, A. (2025). "Symbolic Regression and Misfit-Guided Modeling of B2 Phase Stability in Ru-Containing Refractory Alloys." Acta Materialia. [In Press]

📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

Maintained by the Mahata Lab at Merrimack College.
