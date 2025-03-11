import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, 
    matthews_corrcoef, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, make_scorer
)
from src.collect_data import load_data, save_data

def random_forest_classification():
    data = load_data()

    # Data preparation
    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=22,
        stratify=y
    )

    # Cross-validation
    clf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'f1_macro': 'f1_macro',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'mcc': make_scorer(matthews_corrcoef)
    }

    cv_results = cross_validate(
        estimator=clf_cv,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    cv_accuracy_mean = np.mean(cv_results['test_accuracy'])
    cv_bal_acc_mean = np.mean(cv_results['test_balanced_accuracy'])
    cv_f1_macro_mean = np.mean(cv_results['test_f1_macro'])
    cv_precision_macro_mean = np.mean(cv_results['test_precision_macro'])
    cv_recall_macro_mean = np.mean(cv_results['test_recall_macro'])
    cv_mcc_mean = np.mean(cv_results['test_mcc'])

    print("Cross-Validation Results (5-Fold):")
    print(f"Accuracy (mean):            {cv_accuracy_mean:.3f}")
    print(f"Balanced Accuracy (mean):   {cv_bal_acc_mean:.3f}")
    print(f"Precision Macro (mean):     {cv_precision_macro_mean:.3f}")
    print(f"Recall Macro (mean):        {cv_recall_macro_mean:.3f}")
    print(f"F1 Macro (mean):            {cv_f1_macro_mean:.3f}")
    print(f"MCC (mean):                 {cv_mcc_mean:.3f}")

    # Training on the entire training set
    clf_final = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_final.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = clf_final.predict(X_test)

    # Test set evaluation
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print("\nTest Set Metrics:")
    print(f"Accuracy:            {acc:.3f}")
    print(f"Balanced Accuracy:   {bal_acc:.3f}")
    print(f"Cohen's Kappa:       {kappa:.3f}")
    print(f"Matthews Corrcoef:   {mcc:.3f}")
    print(f"Precision (macro):   {precision_macro:.3f}")
    print(f"Recall (macro):      {recall_macro:.3f}")
    print(f"F1 (macro):          {f1_macro:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=clf_final.classes_
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.show()

    # ROC curves
    y_score = clf_final.predict_proba(X_test)
    class_names = clf_final.classes_
    y_test_bin = label_binarize(y_test, classes=class_names)

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        
        RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc,
            estimator_name=class_name
        ).plot(ax=plt.gca())
        
    plt.title("ROC Curves (One-vs-Rest) - Test Set")
    plt.show()

    roc_auc_macro = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
    print(f"ROC AUC (macro, OvR, test): {roc_auc_macro:.3f}")

def random_forest_ranking():
    combined_df = load_data()
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
    save_data("cross_validation_feature_results_all_features.csv", results_df)
    print("\nCross-validation results saved to 'cross_validation_feature_results.csv'.")