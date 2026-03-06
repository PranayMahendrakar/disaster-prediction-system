# 🌍 Disaster Prediction System

> AI-powered natural disaster prediction using time series models and gradient boosting for flood prediction and earthquake risk assessment.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-red?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit)

---

## 🎯 Overview

The **Disaster Prediction System** is a comprehensive machine learning platform that predicts the risk of natural disasters using state-of-the-art models:

- **🌊 Flood Prediction** — Binary classification of flood events using rainfall, hydrology, and topographic features
- **🏔️ Earthquake Risk** — Multi-class seismic risk assessment (LOW / MEDIUM / HIGH) based on geological and infrastructure data
- **📊 Risk Analytics** — Interactive dashboards with feature importance, risk maps, and time-series forecasts

---

## 🤖 Models

### Flood Prediction

| Model | Type | Description |
|-------|------|-------------|
| **XGBoost** | Gradient Boosting | Optimised for flood binary classification with scale_pos_weight balancing |
| **LightGBM** | Gradient Boosting | Fast gradient boosting with balanced class weights |
| **LSTM** | Time Series (Deep Learning) | Bidirectional LSTM with Conv1D for sequence-based flood prediction |
| **ARIMA/SARIMA** | Time Series (Statistical) | River level forecasting with seasonal patterns |

### Earthquake Risk

| Model | Type | Description |
|-------|------|-------------|
| **XGBoost Classifier** | Gradient Boosting | Multi-class risk classification (LOW/MEDIUM/HIGH) |
| **LightGBM Classifier** | Gradient Boosting | Fast gradient boosting with categorical feature support |
| **XGBoost Regressor** | Gradient Boosting | Continuous risk score prediction (0.0–1.0) |
| **Ensemble** | Weighted Voting | Combines multiple models for robust predictions |

---

## 🌊 Flood Features

| Feature | Description |
|---------|-------------|
| `rainfall_24h_mm` | Rainfall in last 24 hours (mm) |
| `rainfall_72h_mm` | Rainfall in last 72 hours (mm) |
| `rainfall_7day_mm` | Total rainfall over 7 days (mm) |
| `river_level_m` | Current river gauge level (m) |
| `soil_moisture_pct` | Soil moisture percentage |
| `groundwater_level_m` | Groundwater depth (m) |
| `elevation_m` | Site elevation (m) |
| `slope_degrees` | Terrain slope angle (°) |
| `drainage_area_km2` | Watershed drainage area (km²) |
| `temperature_c` | Air temperature (°C) |
| `humidity_pct` | Relative humidity (%) |
| `wind_speed_kmh` | Wind speed (km/h) |

---

## 🏔️ Earthquake Features

| Feature | Description |
|---------|-------------|
| `fault_distance_km` | Distance to nearest fault (km) |
| `fault_type` | Fault mechanism (strike-slip/thrust/normal/oblique) |
| `plate_velocity_mm_yr` | Tectonic plate velocity (mm/yr) |
| `historical_earthquakes_5yr` | Number of earthquakes in last 5 years |
| `last_major_event_years` | Years since last major earthquake |
| `rock_type` | Surface geology type |
| `vs30_m_s` | Shear wave velocity in top 30m (m/s) |
| `soil_amplification_factor` | Ground motion amplification factor |
| `population_density_km2` | Population per km² |
| `building_age_avg_years` | Average building age (years) |
| `seismic_code_compliance` | Fraction of buildings meeting seismic codes |

---

## 📁 Project Structure

```
disaster-prediction-system/
├── app.py                          # Streamlit dashboard (main UI)
├── train.py                        # Training CLI script
├── requirements.txt                # Python dependencies
├── src/
│   ├── data/
│   │   └── data_generator.py       # Synthetic data generation
│   ├── models/
│   │   ├── flood_model.py          # Flood prediction models (XGBoost, LightGBM, LSTM, ARIMA)
│   │   └── earthquake_model.py     # Earthquake risk models (Classifiers, Regressor, Ensemble)
│   └── pipeline/
│       └── prediction_pipeline.py  # End-to-end orchestration pipeline
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train all models
python train.py

# Train flood models only with LightGBM
python train.py --disaster flood --model-type lightgbm

# Train earthquake models with ensemble
python train.py --disaster earthquake --train-ensemble

# Full training with LSTM and ARIMA
python train.py --train-lstm --train-arima --save-models
```

### 3. Launch Dashboard

```bash
streamlit run app.py
```

---

## 🖥️ Dashboard Pages

### 🏠 Home
- System architecture overview
- Model performance comparison chart
- Feature summary for both disaster types

### 🌊 Flood Prediction
- **Predict tab**: Real-time flood risk with interactive sliders for all features
  - Rainfall (24h / 72h / 7-day)
  - River level and soil moisture
  - Topography and weather parameters
- **Model Metrics tab**: Accuracy, F1, ROC-AUC and feature importance
- **Time Series tab**: River level and rainfall visualisation from monitoring stations

### 🏔️ Earthquake Risk
- **Assess Risk tab**: Location + geological input → risk category + score
- **Distribution tab**: Risk score histograms, violin plots by fault type
- **Global Map tab**: Geo-scatter plot of seismic risk worldwide

### 📊 Analytics
- Feature correlation heatmaps
- Regional risk comparisons
- Feature importance rankings

---

## 📊 Model Performance (Expected)

### Flood Models
| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| XGBoost | ~92% | ~91% | ~96% |
| LightGBM | ~91% | ~90% | ~95% |
| LSTM | ~88% | ~87% | ~93% |

### Earthquake Models
| Model | Accuracy | Macro F1 | ROC-AUC |
|-------|----------|----------|---------|
| XGBoost Classifier | ~89% | ~88% | ~95% |
| LightGBM Classifier | ~88% | ~87% | ~94% |
| Risk Regressor (R²) | ~0.88 | — | — |

---

## 🔧 Configuration

### Training CLI Options

```
--disaster {all,flood,earthquake}   Which disaster type to train (default: all)
--model-type {xgboost,lightgbm,...} Gradient boosting model (default: xgboost)
--flood-samples N                   Training samples for flood model (default: 5000)
--earthquake-samples N              Training samples for EQ model (default: 8000)
--train-lstm                        Also train LSTM time series model
--train-arima                       Also train ARIMA river forecaster
--train-ensemble                    Train earthquake ensemble model
--save-models                       Save trained models to disk
--output-json FILE                  Export results to JSON
```

---

## 🏗️ Architecture

```
Input Data ──► Data Generator ──► Feature Engineering
                                           │
                    ┌──────────────────────┤
                    │                      │
              Flood Models          Earthquake Models
                    │                      │
         ┌──────────┤            ┌─────────┤
         │          │            │         │
      XGBoost   LightGBM     XGBoost   LightGBM
      (Boosting) (Boosting)  Classifier  Classifier
         │          │            │         │
      LSTM       ARIMA       XGBoost    Ensemble
     (LSTM)    (Forecaster)  Regressor  (Weighted)
                    │                      │
                    └──────────┬───────────┘
                               │
                    Streamlit Dashboard
                    (Interactive UI + Maps)
```

---

## 📚 Dependencies

- **numpy, pandas** — Data manipulation
- **scikit-learn** — Preprocessing, metrics, pipeline
- **xgboost** — XGBoost gradient boosting
- **lightgbm** — LightGBM gradient boosting
- **tensorflow/keras** — LSTM neural network
- **statsmodels** — ARIMA/SARIMA time series
- **streamlit** — Interactive web dashboard
- **plotly** — Interactive visualizations
- **joblib** — Model serialization

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for disaster risk reduction and early warning systems.*
