import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, make_scorer

# If these imports don't work, ensure your project structure allows them
# (i.e., `src` is recognized as a package).
from src.collect_data import load_data
from src.utils import data_imputation


class RandomForestFeatureRanking:
    """
    Class to perform feature ranking and model evaluation 
    using a RandomForestClassifier on a specified dataset.
    
    Steps:
      1. Load data
      2. Train/Test split
      3. Compute feature importances (Random Forest)
      4. Cross-validate with different numbers of top-k features
      5. Select best k
      6. Evaluate final model on test set
      7. Plot results and save to CSV
    """
    def __init__(self, data_path="all_data.csv", test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

    def run(self):
        # 1) Data Preparation
        combined_df = load_data(self.data_path)
        X = combined_df.drop(columns=["target"])
        y = combined_df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )

        # Ensure columns are strings (sometimes helpful if you have numeric col names)
        X_train.columns = X_train.columns.astype(str)
        X_test.columns  = X_test.columns.astype(str)

        # 2) Feature Importance Ranking
        base_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        base_model.fit(X_train, y_train)

        importances = base_model.feature_importances_
        feature_names = X_train.columns
        indices = np.argsort(importances)[::-1]  # descending
        sorted_features = feature_names[indices]
        sorted_importances = importances[indices]

        print("Feature ranking (top 20):")
        for i in range(min(10, len(sorted_features))):
            print(f"{i+1}. {sorted_features[i]} ({sorted_importances[i]:.4f})")

        # 3) Cross-Validation for Different Feature Counts
        feature_counts = list(range(50, 761, 10))  # 50 to 760 step 10
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        mcc_scorer = make_scorer(matthews_corrcoef)
        scoring = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'mcc': mcc_scorer
        }

        results = []
        for k in feature_counts:
            top_k_features = sorted_features[:k]
            X_train_k = X_train[top_k_features]

            model_k = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            cv_scores = cross_validate(model_k, X_train_k, y_train,
                                       cv=cv, scoring=scoring, n_jobs=-1)

            acc_mean = np.mean(cv_scores['test_accuracy'])
            f1_mean  = np.mean(cv_scores['test_f1_macro'])
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

        # 4) Selecting the Best k (based on F1-macro)
        best_k = results_df.loc[results_df['cv_f1_macro'].idxmax(), 'num_features']
        print(f"\nBest k (based on F1-macro in CV): {best_k}")

        # 5) Train Final Model and Evaluate on Test Set
        final_features = sorted_features[:best_k]
        X_train_final = X_train[final_features]
        X_test_final  = X_test[final_features]

        final_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        final_model.fit(X_train_final, y_train)
        y_pred_test = final_model.predict(X_test_final)

        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1  = f1_score(y_test, y_pred_test, average='macro')
        test_mcc = matthews_corrcoef(y_test, y_pred_test)

        print(f"\n=== Test Set Results for k={best_k} Features ===")
        print(f"Accuracy:  {test_acc:.3f}")
        print(f"F1-macro:  {test_f1:.3f}")
        print(f"MCC:       {test_mcc:.3f}")

        # 6) Cross-Validation Metrics Plot
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

        # 7) Save Cross-Validation Results
        results_df.to_csv("cross_validation_feature_results.csv", index=False)
        print("\nCross-validation results saved to 'cross_validation_feature_results.csv'.")


class RandomForestFeatureRankingImputed:
    """
    Similar to RandomForestFeatureRanking, but loads data from 'all_data.csv' 
    and applies data imputation before the ranking and cross-validation steps.
    """
    def __init__(self, data_path="all_data.csv", test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

    def run(self):
        # 1) Data Preparation with Imputation
        data = load_data(self.data_path)
        data = data_imputation(data)

        X = data.drop(columns=["target"])
        y = data["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )

        # 2) Feature Ranking on Full Dataset
        base_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        base_model.fit(X_train, y_train)

        importances = base_model.feature_importances_
        feature_names = X_train.columns

        indices = np.argsort(importances)[::-1]
        sorted_features = feature_names[indices]
        sorted_importances = importances[indices]

        print("Feature ranking (top 20):")
        for i in range(min(10, len(sorted_features))):
            print(f"{i+1}. {sorted_features[i]} ({sorted_importances[i]:.4f})")

        # 3) Cross-Validation for Different Feature Counts
        feature_counts = [5, 10, 20, 25, 30, 40, 45, 50, 60, 70, 80, 100, 120, 150, 160, 180, 200, 250]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        mcc_scorer = make_scorer(matthews_corrcoef)
        scoring = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'mcc': mcc_scorer
        }

        results = []
        for k in feature_counts:
            top_k_features = sorted_features[:k]
            X_train_k = X_train[top_k_features]

            model_k = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            cv_scores = cross_validate(model_k, X_train_k, y_train,
                                       cv=cv, scoring=scoring, n_jobs=-1)

            acc_mean = np.mean(cv_scores['test_accuracy'])
            f1_mean  = np.mean(cv_scores['test_f1_macro'])
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

        # 4) Selecting the Best k
        best_k = results_df.loc[results_df['cv_f1_macro'].idxmax(), 'num_features']
        print(f"\nBest k (based on F1-macro in CV): {best_k}")

        # 5) Train Final Model and Test
        final_features = sorted_features[:best_k]
        X_train_final = X_train[final_features]
        X_test_final  = X_test[final_features]

        final_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        final_model.fit(X_train_final, y_train)

        y_pred_test = final_model.predict(X_test_final)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1  = f1_score(y_test, y_pred_test, average='macro')
        test_mcc = matthews_corrcoef(y_test, y_pred_test)

        print(f"\n=== Test Set Results for k={best_k} Features ===")
        print(f"Accuracy:  {test_acc:.3f}")
        print(f"F1-macro:  {test_f1:.3f}")
        print(f"MCC:       {test_mcc:.3f}")

        # 6) Cross-Validation Plot
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

        # 7) Save Results
        results_df.to_csv("cross_validation_feature_ranking_results.csv", index=False)
        print("\nCross-validation results saved to 'cross_validation_feature_ranking_results.csv'.")

