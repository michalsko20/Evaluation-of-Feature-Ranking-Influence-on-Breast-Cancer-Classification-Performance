# Feature Ranking & Classification Project

This project focuses on **feature ranking** and **classification** using various machine learning techniques. The primary goal is to determine the best set of features for classification tasks and analyze their impact on model performance.

## 📌 Key Objectives

1. **Data Loading & Preprocessing**  
   - Handling missing values  
   - Standardization & feature scaling  
   - Label encoding for categorical variables  

2. **Feature Ranking Methods**  
   - **Random Forest Importances**  
   - **SVC Coefficients**  
   - **Mutual Information (MI)**  
   - **Permutation Importance**  
   - **LIME-based Explanations**  

3. **Model Training & Cross-Validation**  
   - **Random Forest Classifier**  
   - **Support Vector Classifier (SVC)**  
   - **Multi-Layer Perceptron (MLP) with PyTorch**  
   - **5-fold Cross-Validation** to compare feature subsets  

4. **Visualization & Analysis**  
   - Performance plots (Accuracy, F1, MCC vs. number of features)  
   - Comparison of feature selection methods  
   - LIME-based explainability  

---

## 📁 Project Structure

FEATURE_RANKING
├── data
│   ├── csv_files             # Original CSV files
│   └── processed             # Processed data files
├── models
│   ├── feature_extraction_resnet_and_classification.py
│   ├── MLP_class.py
│   ├── MLP_classification.py
│   ├── random_forest_all_data.py
│   ├── rf_classes.py
│   ├── SVC_rankings.py
│   └── SVC_rankings_classes.py
├── notebooks
│   ├── MLP_visualization.ipynb
│   └── SCV_visualization.ipynb
├── src
│   ├── collect_data.py       # Functions to load/save data
│   ├── feature_ranking.py    # Utilities for ranking features
│   └── utils.py              # Common helper functions (e.g. imputation)
├── cross_validation_feature_ranking_results.csv
├── results.txt
└── README.md                 # This file

---

## ⚙️ How It Works

### **Step 1: Data Preparation**
- Load data from `csv_files/` using `collect_data.py`
- If necessary, apply **data imputation** (`utils.py`)
- Encode categorical values & standardize features (`StandardScaler`)
  
### **Step 2: Feature Ranking**
- Compute **feature importance scores** using different ranking methods.
- The results are stored in CSV files (`cross_validation_feature_ranking_results.csv`).

### **Step 3: Model Training**
- Train classifiers (`models/`) on different ranked feature subsets.
- Run **5-fold cross-validation** (`StratifiedKFold`).

### **Step 4: Evaluation & Visualization**
- Store validation results (`cross_validation_feature_results.csv`).
- Generate **performance plots** using `MLP_visualization.ipynb` or `SVC_visualization.ipynb`.

---

## 🚀 Usage

### **1️⃣ Install Dependencies**
Ensure you have the required Python libraries installed:

```bash
pip install -r requirements.txt
