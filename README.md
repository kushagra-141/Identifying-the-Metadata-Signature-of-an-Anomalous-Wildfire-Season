# Identifying the Metadata Signature of an Anomalous Wildfire Season
## A Big Data Analytics Study Using NASA EONET API v3

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![NASA EONET](https://img.shields.io/badge/Data%20Source-NASA%20EONET%20API%20v3-red)

This project applies big data mining techniques to NASA EONET wildfire metadata to identify structural attributes distinguishing anomalous fire seasons from historical baselines. Using over 11,000 events spanning 2016–2026, it provides a comprehensive analysis of the "Mega-Fire Signature" and introduces three operational fire intelligence tools.

## 🚀 Key Findings
- **Anomalous Volume**: Fire activity in 2024–2026 showed a **23.8×** increase in monthly event counts compared to the 2016–2023 baseline.
- **Geographic Shift**: The geographic center of fire activity moved **1.83 degrees toward the equator** per year ($p = 0.007$).
- **Mega-Fire Signature**: Identified a structural inversion in reporting agency ratios (from IRWIN-dominant to GDACS-dominant) as the strongest predictor of anomalous activity.

## 🛠️ Fire Intelligence System
The project implements three operational tools for wildfire analysis:
1. **Location Fire Risk Scorer**: Divided the globe into $2^\circ \times 2^\circ$ cells to compute 10-year fire recurrence rates (e.g., Los Angeles & Northern California identified as "VERY HIGH" risk).
2. **Monthly Activity Anomaly Detector**: A Decision Tree classifier achieving **96.6% accuracy** ($\pm 1.7\%$) in detecting anomalous months.
3. **Geographic Scope Monitor**: Tracks the equatorward drift and hemisphere distribution of global wildfires.

## 📊 Visualizations
The analysis includes several key visualizations (generated as PNGs):
- **Global Geospatial Map**: Wildfire locations by season.
- **Magnitude Distribution**: KDE and histogram comparisons of fire magnitudes.
- **Anomaly Heatmap**: Event counts by magnitude bin and season.
- **Agency Composition**: The shift between IRWIN and GDACS reporting.

## 💻 Installation & Usage

### Requirements
- Python 3.10+
- Dependencies: `requests`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`

```bash
python -m pip install requests pandas scipy scikit-learn matplotlib seaborn
```

### Execution Order
To reproduce the analysis, run the scripts in the following order:

1. **Fetch Two-Season Data**: `python eonet_fetch.py`
2. **Attribute Discovery**: `python attribute_discovery.py`
3. **Statistical Methodology**: `python methodology.py`
4. **Generate Visualizations**: `python visualizations.py`
5. **Evidence Table**: `python evidence_table.py`
6. **Historical Validation**: `python historical_validation.py`
7. **Fetch Ten-Year Data**: `python fetch_10yr.py` && `python patch_2021.py`
8. **Fire Intelligence System**: `python fire_intelligence.py`

*Note: Data is cached locally after the first fetch. Subsequent runs will use the local CSV files.*

## 📄 Project Structure
- `report.tex`: Full technical report.
- `*.py`: Analysis and data processing scripts.
- `*.csv`: Cached datasets from NASA EONET.
- `*.png`: Generated figures and tools output.

## 👤 Author
**Kushagra Gupta**
- RIT (Rochester Institute of Technology)
- Email: kg2347@g.rit.edu

---
*This study was conducted as a Big Data Analytics project using the NASA EONET API.*
