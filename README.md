# Identifying the Metadata Signature of an Anomalous Wildfire Season

## 🛰️ NASA EONET Big Data Analytics Study

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![NASA EONET](https://img.shields.io/badge/Data%20Source-NASA%20EONET%20API%20v3-red)

### 📌 Project Overview
This project applies big data mining and machine learning techniques to a decade of NASA Earth Observatory Natural Event Tracker (EONET) wildfire metadata (2016–2026). The study identifies a "Mega-Fire Signature"—a structural shift in reporting patterns and geographic distribution that distinguishes the 2024–2026 period as historically anomalous.

### 🚀 Key Findings
- **Volume Surge**: Fire activity in 2024–2026 showed a **23.8×** increase in monthly event counts compared to the 2016–2023 baseline.
- **Reporting Inversion**: Identified a shift from IRWIN-dominant (US domestic) to GDACS-dominant (global) reporting as the strongest metadata predictor of the anomalous period.
- **Equatorward Drift**: Observed a statistically significant southward shift in global fire activity ($1.83^\circ$ per year, $p=0.007$), driven by the expansion of global reporting sources.
- **Magnitude Shift**: Statistical analysis using Robust Z-scores (MAD) confirmed a significant increase in median fire magnitude during the anomaly era.

---

### 🛡️ Fire Intelligence System
The repository implements three operational tools for real-time and historical wildfire analysis:

1.  **📍 Location Fire Risk Scorer**:
    - Divides the globe into $2^\circ \times 2^\circ$ grid cells.
    - Computes 10-year fire recurrence probabilities.
    - Categorizes locations from "LOW" to "VERY HIGH" risk (e.g., Northern California identified as "VERY HIGH").

2.  **📉 Monthly Activity Anomaly Detector**:
    - A Decision Tree classifier trained on metadata features (`log_count`, `irwin_share`, `geographic_diversity`).
    - Achieves **96.6% accuracy** in detecting anomalous fire months.
    - Uses IQR-based fencing for volume-based anomaly detection.

3.  **🌍 Geographic Scope Monitor**:
    - Tracks the hemisphere distribution and mean absolute latitude of global fires.
    - Detects large-scale geographic shifts in fire intelligence reporting.

---

### 📊 Methodology & Visualizations
The analysis is supported by a rigorous statistical framework:
- **Non-Parametric Testing**: Mann-Whitney U tests for magnitude distribution shifts.
- **Robust Statistics**: Median Absolute Deviation (MAD) for z-score normalization in skewed datasets.
- **Visual Evidence**:
    - **Global Geospatial Map**: Visualizing fire intensity by location and season.
    - **Magnitude Distribution**: KDE and log-histogram comparisons.
    - **Anomaly Heatmap**: Stratified analysis of event counts across magnitude bins.
    - **Agency Composition**: Tracking the shift in primary reporting sources.

---

### 💻 Installation & Usage

#### Prerequisites
```bash
python -m pip install requests pandas scipy scikit-learn matplotlib seaborn
```

#### Reproduction Pipeline
Run the scripts in the following order to reproduce the analysis:
1.  `fetch_10yr.py`: Ingests a decade of EONET data month-by-month.
2.  `attribute_discovery.py`: Performs initial feature importance analysis.
3.  `methodology.py`: Applies robust z-scores and trains the classifier.
4.  `visualizations.py`: Generates the primary analysis figures.
5.  `evidence_table.py`: Produces the statistical summary table.
6.  `fire_intelligence.py`: Runs the operational intelligence tools.

---

### 👤 Author
**Kushagra Gupta**  
Rochester Institute of Technology (RIT)  
📧 [kg2347@g.rit.edu](mailto:kg2347@g.rit.edu)

---
*This repository contains the codebase and data analysis for the Big Data Analytics Final Project.*
