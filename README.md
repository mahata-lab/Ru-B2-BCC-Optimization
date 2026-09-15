# Predicting Coherent B2 Stability in Ru-Containing Refractory Alloys

**A physics-guided machine-learning framework combining high-throughput DFT, Random Forest screening, and symbolic regression to design coherent, thermally robust B2 precipitates in refractory BCC matrices.**

This repository accompanies the manuscript *"Predicting Coherent B2 Stability in Ru-Containing Refractory Alloys Through Thermodynamic–Elastic Design Maps"* (A. Mahata, Merrimack College), submitted to *Acta Materialia*. It contains the DFT structural inputs, the processed datasets, the machine-learning and symbolic-regression pipelines, and the scripts and source figures used to produce the paper.

The framework resolves the "binary paradox" — where stoichiometric compounds such as RuHf fail to reach their theoretical solvus temperatures despite high melting points — by treating thermodynamic driving force and lattice compatibility as coupled constraints. The distilled design law is

```
T_solvus (°C) ≈ 0.11 · (ΔH / ΔS_mix) − 20000 · |δ| − 273.15
```

where within the fitted relation a 1% lattice misfit corresponds to roughly a 200 °C penalty.

---

## Repository structure

```
.
├── paper/                     Manuscript, supplementary, bibliography, and final figures
│   ├── manuscript.tex             LaTeX source of the main text
│   ├── supplementary.tex          LaTeX source of the supplementary material
│   ├── references.bib             Bibliography
│   ├── manuscript.pdf             Compiled manuscript (reference copy)
│   ├── supplementary.pdf          Compiled supplementary material
│   ├── figures/                   Final figures as they appear in the paper (PNG)
│   └── figure_scripts/            Scripts that generate the paper figures
│       └── data/                  Projected-DOS and summary data for the figure scripts
│
├── dft/                       First-principles (VASP) inputs and processed data
│   ├── scripts/                   Structure generation and dataset-building scripts
│   ├── input_templates/           INCAR templates (relaxation and static/SCF)
│   ├── structures/                Representative POSCAR inputs, by category
│   │   ├── 01_Baselines/              Elemental BCC references
│   │   ├── 02_Binary_B2/              Ru–X B2 binaries (RuHf, RuTi, RuZr, RuAl)
│   │   ├── 03_Matrix_Binaries/        Binary BCC SQS matrices
│   │   ├── 04_Matrix_Ternaries/       Ternary BCC SQS matrices
│   │   ├── 05_B2_Ternaries_Solubility/ Ternary B2 solubility models
│   │   ├── 06_B2_Penalty_RuSubstitution/ A-site (Ru→Al/Cr/Cu/Si) substitution series
│   │   └── 07_B2_HighOrder/           Higher-order / mixed-B-site B2 structures
│   └── data/                      Processed formation-energy, lattice-parameter, and misfit CSVs
│
└── machine_learning/         Surrogate models, screening, and analysis
    ├── model_comparison/          GPR vs SVR vs RF benchmark (paper Fig. 2)
    ├── screening/                 Final Random Forest screening pipeline and candidate lists
    ├── gridsearch/                Hyperparameter search and cross-validation results
    └── figures/                   ML and grid-search figures
```

---

## Method overview

1. **DFT dataset (`dft/`).** BCC and ordered B2 (CsCl-type) structures are generated with `ase`, and disordered matrices are modelled as special quasirandom structures (SQS) with `icet`. All cells are fully relaxed in VASP (PBE) and followed by a static single-point calculation. Per-atom formation energies are referenced to elemental BCC states, and the coherent misfit `δ = 2(a_B2 − a_BCC)/(a_B2 + a_BCC)` is evaluated for every B2–BCC pair. The `dft/scripts/` build the processed CSVs in `dft/data/`.

2. **Descriptors.** Each configuration is described by DFT quantities (formation-energy contrast, lattice misfit) together with rule-of-mixtures descriptors (VEC, average electronegativity, electronegativity difference, atomic-size mismatch δ_r, Miedema mixing enthalpy).

3. **Surrogate models (`machine_learning/`).** Gaussian process, support vector, and random forest regressors are compared for formation-energy prediction (`model_comparison/`, paper Fig. 2). A tuned dual Random Forest — one surrogate for formation enthalpy, one for lattice misfit — is used to screen the full candidate library (`screening/`), with hyperparameters selected by cross-validated grid search (`gridsearch/`).

4. **Symbolic regression.** The ensemble decision surface is distilled into the closed-form solvus law above, which quantifies the subtractive lattice-strain penalty and defines the coherent, high-stability design window.

---

## Reproducing the figures

The paper figures are produced by the scripts in `paper/figure_scripts/` (structure schematics, model comparison, DFT energetics, lattice misfit, symbolic-regression maps, Al/Si tuning, reference-state benchmark, and electronic structure) and by the ML scripts in `machine_learning/`. Each script reads the CSV/PDOS data included alongside it.

Typical Python dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `ase`, `icet` (and `pysr` for the symbolic-regression stage).

---

## Notes on scope

- **Representative structures.** `dft/structures/` includes a representative POSCAR from each structural category. The full generation recipe for every structure in the study is contained in `dft/scripts/`, and the complete relaxed energetics and lattice parameters are provided in `dft/data/`.
- **Excluded files.** Raw VASP outputs (`OUTCAR`, `CHGCAR`, `WAVECAR`, `vasprun.xml`) and VASP `POTCAR` pseudopotentials are **not** included — `POTCAR` files are distributed under a VASP license and must be obtained from your own licensed VASP installation. Large serialized model binaries are likewise omitted; the models can be regenerated from the scripts and data provided.

---

## Citation

If you use this repository, please cite:

> A. Mahata, *Predicting Coherent B2 Stability in Ru-Containing Refractory Alloys Through Thermodynamic–Elastic Design Maps*, Acta Materialia (submitted).

**Contact:** Avik Mahata — mahataa@merrimack.edu — Department of Mechanical and Electrical Engineering, Merrimack College, North Andover, MA 01845, USA.
