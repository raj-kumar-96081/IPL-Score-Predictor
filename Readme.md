# 🏏 IPL InningSphere: Live Score Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20CatBoost%20%7C%20LightGBM-orange)
![Framework](https://img.shields.io/badge/Frontend-Streamlit-red)

## 📌 Overview
**IPL InningSphere** is a machine learning-powered web application that forecasts the final first-innings score of an Indian Premier League (IPL) match in real-time. 

Built with a robust data pipeline processing over **150,000+ historical ball-by-ball and match-level records**, the system dynamically computes context-aware features such as instantaneous run rates, resource availability, and custom pressure indices to deliver highly accurate predictions.

---

## 🚀 Key Features
* **Real-Time Feature Engineering:** The inference pipeline instantly calculates `balls_left`, `wickets_left`, and `current_run_rate` (CRR) from live user inputs.
* **Context-Aware Metrics:** Captures match momentum through custom variables like runs scored in the last 5 overs and a Wicket Pressure Index.
* **Ensemble Model Selection:** Users can dynamically toggle between 5 distinct trained models directly from the UI:
  * XGBoost Regressor
  * CatBoost Regressor
  * LightGBM Regressor
  * Random Forest
  * Voting Regressor (Ensemble)
* **Interactive UI:** A fully responsive, custom-styled Streamlit dashboard featuring local background integration and intuitive controls.

---

## 📊 Model Performance & Evaluation
A broad spectrum of algorithms was evaluated, ranging from classical Linear Regression to state-of-the-art tree-based ensembles. 

The primary model (**XGBoost Regressor**) was optimized using second-order gradients and achieved the following metrics on the test dataset:
* **R² Score:** `0.978` (Exceptional variance explanation)
* **Mean Absolute Error (MAE):** `± 2.1 runs`

---

## 🛠️ Technology Stack
* **Language:** Python
* **Data Processing & EDA:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost, CatBoost, LightGBM
* **Model Serialization:** Pickle
* **Web Framework:** Streamlit
* **Deployment:** Local / External Cloud Host

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/raj-kumar-96081/IPL-Score-Predictor](https://github.com/raj-kumar-96081/IPL-Score-Predictor)
cd IPL-InningSphere

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required packages using:

```bash
pip install -r requirements.txt