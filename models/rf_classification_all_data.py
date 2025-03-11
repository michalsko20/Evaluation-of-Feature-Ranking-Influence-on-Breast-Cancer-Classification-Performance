import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, make_scorer
from src.collect_data import load_data
def rf_classification():
    combined_df = load_data("all_data.csv")
    # --------------------------------------------------
    # 1) Data Preparation
    # --------------------------------------------------
    # Assume 'combined_df' is a DataFrame where 'target' is the label column.
    X = combined_df.drop(columns=["target"])
    y = combined_df["target"]

    # Split data into training and testing sets (e.g., 80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # --------------------------------------------------
    # 2) Feature Importance Ranking
    # --------------------------------------------------
    # Train a Random Forest model to compute feature importance.
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)

    # Extract feature importances and sort them in descending order.
    importances = base_model.feature_importances_
    feature_names = X_train.columns
    indices = np.argsort(importances)[::-1]
    sorted_features = feature_names[indices]
    sorted_importances = importances[indices]

    # Display the top 20 features.
    print("Feature ranking (top 20):")
    for i in range(min(10, len(sorted_features))):
        print(f"{i+1}. {sorted_features[i]} ({sorted_importances[i]:.4f})")

    # --------------------------------------------------
    # 3) Cross-Validation for Different Feature Counts
    # --------------------------------------------------
    # Define different numbers of features (k) to test.
    feature_counts = list(range(50, 761, 10))

    # Set up 5-fold stratified cross-validation.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define scoring metrics for cross-validation.
    mcc_scorer = make_scorer(matthews_corrcoef)
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'mcc': mcc_scorer
    }

    # Perform cross-validation for different feature counts.
    results = []
    for k in feature_counts:
        top_k_features = sorted_features[:k]

        # Create a subset with the top-k features.
        X_train_k = X_train[top_k_features]

        # Train a Random Forest model with the selected features.
        model_k = RandomForestClassifier(n_estimators=100, random_state=42)

        # Perform cross-validation.
        cv_scores = cross_validate(
            model_k, X_train_k, y_train,
            cv=cv, scoring=scoring, n_jobs=-1
        )

        # Record the average metrics from cross-validation.
        acc_mean = np.mean(cv_scores['test_accuracy'])
        f1_mean = np.mean(cv_scores['test_f1_macro'])
        mcc_mean = np.mean(cv_scores['test_mcc'])

        results.append({
            'num_features': k,
            'cv_accuracy': acc_mean,
            'cv_f1_macro': f1_mean,
            'cv_mcc': mcc_mean
        })

    results_df = pd.DataFrame(results)
    print("\nCross-validation results (averages) for different feature counts:")
    print(results_df)

    # --------------------------------------------------
    # 4) Selecting the Best k
    # --------------------------------------------------
    # Select the best k based on the highest F1-macro score in cross-validation.
    best_k = results_df.loc[results_df['cv_f1_macro'].idxmax(), 'num_features']
    print(f"\nBest k (based on F1-macro in CV): {best_k}")

    # --------------------------------------------------
    # 5) Train Final Model and Evaluate on Test Set
    # --------------------------------------------------
    # Create a feature subset with the top-best_k features.
    final_features = sorted_features[:best_k]
    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]

    # Train the final Random Forest model.
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X_train_final, y_train)

    # Evaluate the model on the test set.
    y_pred_test = final_model.predict(X_test_final)

    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    test_mcc = matthews_corrcoef(y_test, y_pred_test)

    print(f"\n=== Test Set Results for k={best_k} Features ===")
    print(f"Accuracy:  {test_acc:.3f}")
    print(f"F1-macro:  {test_f1:.3f}")
    print(f"MCC:       {test_mcc:.3f}")

    # --------------------------------------------------
    # 6) Cross-Validation Metrics Plot (Accuracy, F1, MCC vs k)
    # --------------------------------------------------
    # Plot metrics from cross-validation for different feature counts.
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['num_features'], results_df['cv_accuracy'], marker='o', label='CV Accuracy')
    plt.plot(results_df['num_features'], results_df['cv_f1_macro'], marker='s', label='CV F1 Macro')
    plt.plot(results_df['num_features'], results_df['cv_mcc'], marker='^', label='CV MCC')
    plt.xlabel('Number of Selected Features (k)')
    plt.ylabel('Metrics (CV Averages)')
    plt.title('Cross-Validation Results vs Number of Selected Features')
    plt.grid(True)
    plt.legend()
    plt.show()

    # --------------------------------------------------
    # 7) Save Cross-Validation Results to CSV
    # --------------------------------------------------
    # Save the cross-validation results to a CSV file.
    results_df.to_csv("cross_validation_feature_results.csv", index=False)
    print("\nCross-validation results saved to 'cross_validation_feature_results.csv'.")

def rf_classification_G1_imputated():
    from src.utils import data_imputation
    data = load_data("final_features.csv")
    data = data_imputation(data)
    # ========= 1) Data Preparation =========
    # Assume 'data' is a DataFrame with no missing values and 'target' is the target label column
    X = data.drop(columns=["target"])
    y = data["target"]

    # Split data into training and testing sets (e.g., 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ========= 2) Feature Ranking on Full Dataset =========
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)

    importances = base_model.feature_importances_
    feature_names = X_train.columns

    # Sort features by importance in descending order
    indices = np.argsort(importances)[::-1]
    sorted_features = feature_names[indices]
    sorted_importances = importances[indices]

    print("Feature ranking (top 20):")
    for i in range(min(10, len(sorted_features))):
        print(f"{i+1}. {sorted_features[i]} ({sorted_importances[i]:.4f})")

    # ========= 3) Cross-Validation for Different Feature Counts =========
    # Define different feature counts to test
    feature_counts = [5, 10, 20, 25, 30, 40, 45, 50, 60, 70, 80, 100, 120, 150, 160, 180, 200, 250]

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define scorers for cross-validation
    mcc_scorer = make_scorer(matthews_corrcoef)
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'mcc': mcc_scorer
    }

    results = []

    for k in feature_counts:
        top_k_features = sorted_features[:k]

        # Create a subset with top-k features
        X_train_k = X_train[top_k_features]

        # Define the model
        model_k = RandomForestClassifier(n_estimators=100, random_state=42)

        # Perform cross-validation
        cv_scores = cross_validate(
            model_k, X_train_k, y_train,
            cv=cv, scoring=scoring, n_jobs=-1
        )

        acc_mean = np.mean(cv_scores['test_accuracy'])
        f1_mean = np.mean(cv_scores['test_f1_macro'])
        mcc_mean = np.mean(cv_scores['test_mcc'])

        results.append({
            'num_features': k,
            'cv_accuracy': acc_mean,
            'cv_f1_macro': f1_mean,
            'cv_mcc': mcc_mean
        })

    results_df = pd.DataFrame(results)
    print("\nCross-validation results (averages) for different feature counts:")
    print(results_df)

    # ========= 4) Selecting the Best k =========
    # Example: Select k based on the highest F1-macro score in cross-validation
    best_k = results_df.loc[results_df['cv_f1_macro'].idxmax(), 'num_features']
    print(f"\nBest k (based on F1-macro in CV): {best_k}")

    # ========= 5) Train Final Model and Test ============
    # Create final subset of features (top-best_k) on the full training set
    final_features = sorted_features[:best_k]
    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]

    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X_train_final, y_train)

    y_pred_test = final_model.predict(X_test_final)

    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    test_mcc = matthews_corrcoef(y_test, y_pred_test)

    print(f"\n=== Test Set Results for k={best_k} Features ===")
    print(f"Accuracy:  {test_acc:.3f}")
    print(f"F1-macro:  {test_f1:.3f}")
    print(f"MCC:       {test_mcc:.3f}")

    # ========= 6) Cross-Validation Plot (Accuracy, F1, MCC vs k) ============
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['num_features'], results_df['cv_accuracy'], marker='o', label='CV Accuracy')
    plt.plot(results_df['num_features'], results_df['cv_f1_macro'], marker='s', label='CV F1 Macro')
    plt.plot(results_df['num_features'], results_df['cv_mcc'], marker='^', label='CV MCC')
    plt.xlabel('Number of Selected Features (k)')
    plt.ylabel('Metrics (CV Averages)')
    plt.title('Cross-Validation Results vs Number of Selected Features')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ========= 7) Save Cross-Validation Results to CSV ==========
    results_df.to_csv("cross_validation_feature_ranking_results.csv", index=False)
    print("\nCross-validation results saved to 'cross_validation_feature_ranking_results.csv'.")
