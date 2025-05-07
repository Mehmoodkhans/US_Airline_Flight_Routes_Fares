
# ✈️ Airline Fare Prediction

## 📌 Overview
This project focuses on predicting average airline fares between origin and destination city pairs using a comprehensive dataset of U.S. flight routes, carriers, and fare data. The goal is to build accurate regression models using engineered features derived from flight metrics, geographic distances, and market performance data across nearly three decades (1993–2024).

## 📂 About the Dataset
**Source**: [Kaggle – US Airline Flight Routes and Fares (1993–2024)](https://www.kaggle.com/datasets/bhavikjikadara/us-airline-flight-routes-and-fares-1993-2024/data)

US_routes_fare/
├── data/
│   └── US Airline Flight Routes and Fares 1993-2024.zip
├── notebook/
│   └── US_flights_main.ipynb
├── README.md

This dataset provides detailed information on airline flight routes, fares, and passenger volumes within the United States 
from 1993 to 2024. It includes:
- Origin and destination cities and airport IDs
- Geographical distances between airports
- Number of passengers
- Fare-related details by the largest and lowest fare carriers

### 🔧 Potential Uses:
- Market analysis & trend discovery  
- Airline fare prediction & optimization  
- Competitive strategy & route planning  
- Economic and behavioral studies on travel  
- Geospatial and temporal analysis

## 🧭 Notebook Overview: How It Works

This notebook follows a structured four-part pipeline to explore, clean, and model airline fare data. It combines data intuition with practical machine learning steps, and is designed to support both analysis and reproducibility.

### 🟩 Part A: Introduction & Setup
- Imported core **libraries** (`pandas`, `numpy`, visual tools)
- Loaded the **airline dataset**
- Performed basic checks to understand **shape, columns, and data types**

### 📊 Part B: Exploratory Data Analysis (EDA)
- Explored **distributions** and missing values
- Visualized and analyzed **key features**
- Identified **hidden patterns** and the target variable (`fare`)
- Gained insight into relationships between variables (e.g., coordinates, carriers, costs)

### 🧹 Part C: Data Preprocessing
- Cleaned missing geocoordinates using **mean city-based imputation**
- Removed **rows with fully missing carrier/fare data**
- Transformed or encoded **categorical columns**
- Dropped **redundant features** to reduce noise
- Prepared the dataset for both **linear and tree-based modeling**

### 🌡️ Part D: Modeling & Evaluation
- Applied **Linear Models**:  
  - `LinearRegression`, `RidgeCV`, `LassoCV`, `ElasticNetCV`
  - Analyzed **coefficients** to interpret feature influence  
- Applied **Tree-based Models**:  
  - `RandomForestRegressor`, `XGBoost`, `CatBoost`, `LightGBM`, `GradientBoosting`, `ExtraTrees`
  - Used **feature importance** to interpret the models

### 🏹 Part E: Cross Validation
- Applied **K-Fold Cross-Validation** to evaluate model consistency
- Compared model performance using metrics:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)
  - **R² Score** (Explained Variance)

## 🧰 Tech Stack
- **Language**: Python 3.x  
- **Core Libraries**:  
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `re`, `ast`, `warnings`, `zipfile`  
- **Preprocessing**:  
  `category_encoders`, `LabelEncoder`, `StandardScaler`  
- **Modeling**:  
  `scikit-learn`, `xgboost`, `lightgbm`, `catboost`  

## 📈 Results Summary

- Tree-based models generally outperformed linear models in terms of RMSE and R².
- `HistGradientBoosting` and `XGBoost` were top performers after cross-validation, showing both high predictive accuracy and generalization ability.
- Feature importance revealed that distance and fare-specific features (e.g., `fare_lg`, `fare_low`) are consistently the most influential across models.
- Notably, linear models also showed consistent feature importance across regularized variants (Ridge, Lasso, ElasticNet), indicating robust data preparation and collinearity handling.

## 🔍 Model Evaluation: Linear Models

| Feature            | Linear | Ridge  | Lasso  | ElasticNet |
|--------------------|--------|--------|--------|------------|
| fare_lg            | 54.52  | 54.50  | 54.52  | 54.50      |
| fare_low           | 22.08  | 22.09  | 22.08  | 22.09      |
| nsmiles            | 4.64   | 4.64   | 4.64   | 4.64       |
| lf_ms              | -2.68  | -2.69  | -2.68  | -2.68      |
| passengers         | -2.45  | -2.45  | -2.45  | -2.45      |

> 💡 Interpretation: The consistency in feature weights across linear models (especially regularized ones) validates the stability of the data and highlights the dominant predictive strength of `fare_lg` and `fare_low`.

## 🌲 Feature Importance: Tree-Based Models

| Feature           | CatBoost | LightGBM | XGBoost | RandomForest | ExtraTrees | GradientBoosting |
|------------------|----------|----------|---------|---------------|-------------|------------------|
| fare_lg          | 60.52    | 18.23    | 15.41   | **90.11**     | 41.29       | **87.39**        |
| fare_low         | 18.84    | 12.87    | 10.89   | 4.76          | 35.37       | 10.83            |
| nsmiles          | 3.03     | 10.50    | 9.27    | 0.68          | 16.64       | 0.46             |
| large_ms         | 4.20     | 12.47    | 13.79   | 1.20          | 1.02        | 0.26             |
| passengers       | 2.98     | 10.53    | 10.50   | 0.72          | 0.57        | 0.28             |

> 📌 Interpretation: While Random Forest placed 90% of importance on `fare_lg`, boosting models like LightGBM and XGBoost distributed importance more broadly across multiple features — highlighting the trade-off between interpretability and model complexity.

## ✅ Cross-Validation Results (Tree Models)

| Model                   | Mean RMSE | Std RMSE | Mean R² | Std R² |
|-------------------------|-----------|----------|---------|--------|
| Random Forest           | **15.89** | 1.17     | **0.960** | 0.0057 |
| CatBoost                | 17.48     | 1.55     | 0.9515 | 0.0085 |
| LightGBM                | 17.81     | 1.62     | 0.9497 | 0.0091 |
| HistGradientBoosting    | 17.84     | 1.57     | 0.9495 | 0.0088 |
| XGBoost                 | 17.85     | 1.61     | 0.9494 | 0.0091 |

> ⚖️ Summary: Random Forest shows the lowest RMSE and highest R² but is heavily dependent on one feature. In contrast, boosting models provide a more balanced view of feature contributions, making them favorable in scenarios prioritizing model interpretability and fairness across variables.


## 📎 File Structure
- `1_data_loading_and_eda.ipynb` – Data import, inspection, EDA  
- `2_feature_engineering_and_cleaning.ipynb` – Data cleaning, encoding, transformation  
- `3_modeling_linear_models.ipynb` – Linear regression models and analysis  
- `4_modeling_tree_models.ipynb` – Advanced tree-based and boosting models  
- `5_cross_validation_and_comparison.ipynb` – Model comparison and final evaluation  

## ✅ How to Run
1. Clone the repository or download the notebooks.
2. Install required packages via:
   ```bash
   pip install -r requirements.txt
   ```
3. Start from `1_data_loading_and_eda.ipynb` and follow through the pipeline sequentially.

## 📬 Contact
For any feedback or collaboration inquiries, feel free to reach out via GitHub or Kaggle.
