# Lead Price Prediction: Macroeconomic Impact Analysis

This project aims to predict 7-day returns of London Metal Exchange (LME) lead prices and analyze the impact of macroeconomic factors on lead price movements. The project includes a complete machine learning workflow from data preprocessing and feature engineering to model training and evaluation.

## 📋 Project Overview

This project employs various machine learning models (linear models, tree-based models, deep learning models, and time series models) to predict 7-day returns of lead prices, performing both regression and classification tasks. The project also incorporates macroeconomic factor analysis, including copper prices, US Dollar Index, US 10-year Treasury yield, and WTI crude oil prices.

### Main Tasks

1. **Regression Task**: Predict continuous 7-day forward returns (return_7d) of lead prices
2. **Classification Task**: Predict the direction of 7-day returns (positive/negative)

## 📁 Project Structure

```
Project_Macro_Impact_On_Lead/
├── data_raw/                    # Raw data files
│   ├── LOPBDY LME Comdty.xlsx  # LME lead spot price
│   ├── LMPBDS03 LME Comdty.xlsx # LME lead 3-month futures price
│   ├── all_raw_data.csv        # Aligned raw data (with macro factors)
│   ├── copper_HG_F_raw.csv     # Copper futures price
│   ├── dxy_usd_index.csv       # US Dollar Index
│   ├── us_10yr_yield_raw.csv   # US 10-year Treasury yield
│   └── wti_crude_oil_CLF.csv   # WTI crude oil price
│
├── data_processed/              # Processed data files
│   ├── lopbdy_features.csv     # Pure price features (no macro factors)
│   ├── all_features.csv        # All features (with macro factors)
│   └── macro_features.csv      # Macroeconomic features
│
├── eda/                         # Exploratory Data Analysis
│   ├── EDA.ipynb               # Initial EDA (all_features)
│   └── EDA_lopbdy.ipynb        # LOPBDY features EDA
│
├── notebooks/                   # Model training and comparison
│   ├── model_comparision_regression.ipynb    # Regression model comparison
│   └── model_comparision_classification.ipynb # Classification model comparison
│
├── src/                         # Source code
│   ├── features/                # Feature engineering
│   │   ├── feature_engineering.py      # Technical indicator features
│   │   ├── macro_feature_engineering.py # Macroeconomic factor features
│   │   └── all_features.py             # Feature integration
│   │
│   ├── regression/              # Regression models
│   │   ├── reg_baseline_linear.py     # Linear Regression, Ridge Regression
│   │   ├── reg_tree_models.py          # Random Forest, XGBoost
│   │   ├── reg_lstm.py                 # LSTM model
│   │   ├── reg_transformer.py          # Transformer model
│   │   └── reg_time_series.py         # ARIMA, SARIMA models
│   │
│   ├── classification/          # Classification models
│   │   ├── clf_baseline_linear.py      # Logistic Regression, Ridge Classification
│   │   ├── clf_tree_models.py          # Random Forest
│   │   ├── clf_svm_knn.py              # SVM, KNN
│   │   ├── clf_mlp.py                  # Multi-Layer Perceptron
│   │   └── clf_rnn.py                   # RNN (LSTM)
│   │
│   ├── data/                    # Data processing
│   │   └── train_test_split_walk_forward.py # Time series data splitting
│   │
│   └── evaluation/              # Model evaluation utilities
│
├── report/                      # Report files
├── presentation/                # Presentation materials
└── README.md                    # Project documentation
```

## 🚀 Quick Start

### Requirements

- Python 3.8+
- Jupyter Notebook
- Key dependencies:
  - pandas, numpy
  - scikit-learn
  - xgboost
  - tensorflow/keras (for deep learning models)
  - statsmodels (for ARIMA/SARIMA)
  - matplotlib, seaborn (for visualization)

### Installation

1. **Clone the repository**
```bash
git clone <repository_url>
cd Project_Macro_Impact_On_Lead
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare data**
   - Place raw data files in the `data_raw/` directory
   - Run feature engineering scripts to generate processed data

4. **Run feature engineering**
```bash
python src/features/feature_engineering.py
```

## 📊 Data Description

### Data Sources

- **LOPBDY**: LME lead spot price (primary data source)
- **LMPBDS03**: LME lead 3-month futures price
- **Macroeconomic Factors**:
  - Copper futures price (HG-F)
  - US Dollar Index (DXY)
  - US 10-year Treasury yield
  - WTI crude oil price

### Data Time Range

- Start date: December 2000
- End date: October 2025
- Total observations: ~6,149 trading days (after alignment)

### Feature Description

**Price-based Technical Indicators** (14 features):
- Moving Averages: MA7, MA30
- Exponential Moving Averages: EMA5, EMA15, EMA30
- MACD Indicators: MACD5, MACD15, MACD30
- RSI (Relative Strength Index)
- Bollinger Bands: BB_upper, BB_middle, BB_lower
- Returns: returns (daily returns)
- Rolling Volatility: rolling_volatility

**Target Variables**:
- `return_7d`: 7-day forward return (regression target)
- `target`: Binary classification label (1 if return_7d > 0, else 0)

## 🤖 Models

### Regression Models

1. **Linear Models**
   - Linear Regression
   - Ridge Regression (with cross-validation)

2. **Tree-based Models**
   - Random Forest (100 trees)
   - XGBoost

3. **Deep Learning Models**
   - LSTM (sequence length=10, two LSTM layers [64, 32])
   - Transformer (d_model=64, num_heads=4, num_layers=2)

4. **Time Series Models**
   - ARIMA (auto parameter selection)
   - SARIMA (seasonal ARIMA, auto parameter selection)

### Classification Models

1. **Linear Models**
   - Logistic Regression
   - Ridge Classification

2. **Tree-based Models**
   - Random Forest

3. **Traditional Machine Learning**
   - SVM (Support Vector Machine)
   - KNN (K-Nearest Neighbors)

4. **Neural Networks**
   - MLP (Multi-Layer Perceptron)
   - RNN (LSTM)

### Evaluation Metrics

**Regression Task**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

**Classification Task**:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

## 📝 Usage

### 1. Data Preprocessing

Run feature engineering to generate features:
```python
from src.features.feature_engineering import process_lopbdy_features

# Process LOPBDY features
df = process_lopbdy_features(
    data_path="data_raw/LOPBDY LME Comdty.xlsx",
    save_path="data_processed/lopbdy_features.csv"
)
```

### 2. Exploratory Data Analysis

Open and run `eda/EDA_lopbdy.ipynb` for data exploration:
- Descriptive statistics
- Correlation analysis
- PCA (Principal Component Analysis)
- VIF (Variance Inflation Factor) analysis

### 3. Model Training

**Regression Task**:
Open and run `notebooks/model_comparision_regression.ipynb`

**Classification Task**:
Open and run `notebooks/model_comparision_classification.ipynb`

### 4. Feature Selection

Current baseline model uses 5 low-correlation features:
- MACD30
- MACD15
- RSI
- returns
- rolling_volatility

## 🔍 Key Findings

1. **Data Quality**: Using pure price features (`lopbdy_features.csv`) avoids data leakage issues
2. **Feature Selection**: Optimal feature sets selected through PCA, correlation analysis, and VIF analysis
3. **Model Performance**: Different model types show significant performance differences across tasks
4. **Time Series Characteristics**: Walk-forward validation ensures proper time series data splitting

## 📈 Results

Model results are saved in their respective notebooks, including:
- Performance metrics comparison across all models
- Visualization charts (RMSE/MAE comparison, ROC curves, confusion matrices, etc.)
- Training time comparison
- Best model identification

## ⚠️ Important Notes

1. **Data Leakage Prevention**:
   - Uses walk-forward time series splitting
   - Ensures features only use historical information
   - Avoids using future information (e.g., future_return_7d)

2. **Feature Engineering**:
   - All technical indicators calculated based on historical prices
   - Target variable (return_7d) calculated using 7-day forward prices

3. **Model Training**:
   - All models evaluated on the same test set
   - Standardization applied (for linear models and neural networks)
   - Tree models do not require feature scaling

## 📚 File Descriptions

### Core Files

- `src/features/feature_engineering.py`: Technical indicator feature engineering
- `src/regression/`: All regression model implementations
- `src/classification/`: All classification model implementations
- `notebooks/model_comparision_*.ipynb`: Model training and comparison notebooks

### Data Files

- `data_processed/lopbdy_features.csv`: Pure price features (recommended for baseline models)
- `data_processed/all_features.csv`: Complete feature set including macroeconomic factors
- `data_raw/all_raw_data.csv`: Aligned raw data with macroeconomic factors

## 🔧 Development

### Adding New Models

1. Create a new file in the corresponding `src/regression/` or `src/classification/` directory
2. Implement model training function that returns a dictionary with predictions and evaluation metrics
3. Import and call the new model in the notebook

### Modifying Feature Sets

Modify the `feature_cols` variable in the notebook:
```python
feature_cols = ['MACD30', 'MACD15', 'RSI', 'returns', 'rolling_volatility']
```

## 📄 License

[Add license information]

## 👥 Authors

[Add author information]

## 🙏 Acknowledgments

[Add acknowledgments]
