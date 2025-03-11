import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, make_scorer
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def SVC_rankings():
    from src.utils import data_imputation
    from src.collect_data import load_data
    # ================== 1) Data Loading & Splitting ==================
    data = load_data("final_features.csv")
    data = data_imputation(data)

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ================== 2) Functions to Determine Rankings ==================

    def get_ranking_filter_mi(X_tr, y_tr):
        """Returns a list of features sorted by mutual_info_classif (descending)."""
        mi_values = mutual_info_classif(X_tr, y_tr, discrete_features=False, random_state=42)
        mi_indices = np.argsort(mi_values)[::-1]
        return X_tr.columns[mi_indices]

    def get_ranking_svc_coef(X_tr, y_tr):
        """Returns a list of features sorted by |coef_| from SVC (kernel='linear'), with standardization."""
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel='linear', max_iter=20000, random_state=42))
        ])
        pipe.fit(X_tr, y_tr)
        svc_linear = pipe.named_steps["svc"]
        importances = np.abs(svc_linear.coef_).mean(axis=0)  # if multi-class: take the mean
        indices = np.argsort(importances)[::-1]
        return X_tr.columns[indices]

    def get_ranking_rfe(X_tr, y_tr):
        """Returns a list of features ranked using RFE with SVC (kernel='linear'), with standardization."""
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        svc_lin = SVC(kernel='linear', max_iter=20000, random_state=42)
        rfe = RFE(estimator=svc_lin, n_features_to_select=1, step=1)
        rfe.fit(X_tr_scaled, y_tr)
        indices = np.argsort(rfe.ranking_)
        return X_tr.columns[indices]

    def get_ranking_permutation(X_tr, y_tr):
        """Returns a list of features ranked by accuracy drop upon permutation (SVC RBF)."""
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel='rbf', random_state=42))
        ])
        pipe.fit(X_tr, y_tr)
        X_tr_scaled = pipe.named_steps["scaler"].transform(X_tr)
        svc_rbf = pipe.named_steps["svc"]
        perm = permutation_importance(svc_rbf, X_tr_scaled, y_tr, scoring="accuracy",
                                    n_repeats=5, random_state=42)
        importances = perm.importances_mean
        indices = np.argsort(importances)[::-1]
        return X_tr.columns[indices]

    # ================== 3) Compute Feature Rankings ==================
    mi_sorted   = get_ranking_filter_mi(X_train, y_train)
    svc_sorted  = get_ranking_svc_coef(X_train, y_train)
    rfe_sorted  = get_ranking_rfe(X_train, y_train)
    perm_sorted = get_ranking_permutation(X_train, y_train)

    ranking_dict = {
        "Filter_MI": mi_sorted,
        "SVC_coef" : svc_sorted,
        "RFE"      : rfe_sorted,
        "PermImp"  : perm_sorted
    }

    # ================== 4) Cross-Validation for Different Rankings and k ==================
    feature_counts = [5, 10, 20, 30, 40, 50, 60, 80, 100]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mcc_scorer = make_scorer(matthews_corrcoef)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "mcc": mcc_scorer}

    all_results = []

    for rank_name, sorted_feats in ranking_dict.items():
        print(f"\n=== RANKING: {rank_name} ===")
        results = []

        for k in feature_counts:
            top_k = sorted_feats[:k]
            
            # Pipeline: Standardization + SVC(RBF)
            model_k = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel='rbf', random_state=42))
            ])
            
            # Cross-validation
            cv_scores = cross_validate(
                model_k, X_train[top_k], y_train,
                cv=cv, scoring=scoring, n_jobs=-1
            )
            
            acc_mean = np.mean(cv_scores["test_accuracy"])
            f1_mean  = np.mean(cv_scores["test_f1_macro"])
            mcc_mean = np.mean(cv_scores["test_mcc"])
            
            results.append({
                "ranking": rank_name,
                "num_features": k,
                "cv_accuracy": acc_mean,
                "cv_f1_macro": f1_mean,
                "cv_mcc": mcc_mean
            })
        
        df_r = pd.DataFrame(results)
        print(df_r)
        
        # Find best_k based on f1_macro
        best_idx = df_r["cv_f1_macro"].idxmax()
        best_k = df_r.loc[best_idx, "num_features"]
        print(f"Best k for {rank_name}: {best_k}")

        # Train final model (without tuning) on best_k
        final_feats = sorted_feats[:best_k]
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel='rbf', random_state=42))
        ])
        final_model.fit(X_train[final_feats], y_train)
        
        # Test evaluation
        y_pred_test = final_model.predict(X_test[final_feats])
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average="macro")
        test_mcc = matthews_corrcoef(y_test, y_pred_test)
        print(f"[{rank_name}] Test results for k={best_k} => "
              f"ACC={test_acc:.3f}, F1={test_f1:.3f}, MCC={test_mcc:.3f}")

        all_results.extend(results)

    # Convert results to DataFrame
    all_results_df = pd.DataFrame(all_results)

    # ================== 5) Hyperparameter Tuning (for each ranking at best_k) ==================
    param_grid = {
        "svc__C":     [0.01, 0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1]
    }
    tuned_results = []

    for rank_name in ranking_dict.keys():
        best_k = all_results_df[all_results_df["ranking"] == rank_name]["num_features"].max()
        sorted_feats = ranking_dict[rank_name]
        final_feats = sorted_feats[:best_k]

        print(f"\n>>> Tuning hyperparams for ranking={rank_name}, k={best_k}")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel='rbf', random_state=42))
        ])
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1
        )
        grid.fit(X_train[final_feats], y_train)
        
        print(f"Best params for {rank_name}: {grid.best_params_}")
        print(f"Best CV score (f1_macro) for {rank_name}: {grid.best_score_:.4f}")

        best_model = grid.best_estimator_
        y_pred_test = best_model.predict(X_test[final_feats])

        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average="macro")
        test_mcc = matthews_corrcoef(y_test, y_pred_test)

        tuned_results.append({
            "ranking": rank_name,
            "best_k": best_k,
            "best_params": grid.best_params_,
            "test_acc_tuned": test_acc,
            "test_f1_macro_tuned": test_f1,
            "test_mcc_tuned": test_mcc
        })

    tuned_df = pd.DataFrame(tuned_results)
    print("\n=== Tuned results for each ranking ===")
    print(tuned_df)


def PCA_SHAP_SVC_ranking():
    from src.utils import data_imputation
    from src.collect_data import load_data
    import shap
    
    data = load_data("final_features.csv")
    data = data_imputation(data)
    
    # ========== 1) Data Preparation ==========
    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ========== 2) Pipeline PCA + SVC (probability=True for SHAP) ==========
    pca_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.80)),  # e.g., 80% explained variance
        ("svc", SVC(kernel='rbf', probability=True, random_state=42))
    ])

    # Train the pipeline
    pca_pipeline.fit(X_train, y_train)

    scaler_ = pca_pipeline.named_steps["scaler"]
    pca_    = pca_pipeline.named_steps["pca"]
    svc_    = pca_pipeline.named_steps["svc"]

    print("Number of components after PCA =", pca_.n_components_)

    # ========== 3) SHAP in PCA space ==========
    def custom_predict_proba_pca_only(X_):
        """Assumes X_ is already in PCA space. Only SVC(prob=True).predict_proba(X_)."""
        return svc_.predict_proba(X_)

    n_bg = min(100, len(X_train))
    n_subset = min(200, len(X_train))

    X_bg = X_train.sample(n_bg, random_state=42)
    X_sub= X_train.sample(n_subset, random_state=123)

    # Manually scale and apply PCA
    X_bg_pca = pca_.transform(scaler_.transform(X_bg))
    X_sub_pca= pca_.transform(scaler_.transform(X_sub))

    explainer = shap.KernelExplainer(
        model=custom_predict_proba_pca_only,
        data=X_bg_pca,
        link="logit"
    )

    shap_values = explainer.shap_values(
        X_sub_pca,
        nsamples=100,
        l1_reg="num_features(10)"
    )

    # Handling multi-class case
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # list [array_class1, ..., array_classN] => (n_samples, m)
        abs_shaps = [np.abs(sv) for sv in shap_values]
        mean_abs_shaps = np.mean(abs_shaps, axis=0)   # => (n_samples, m)
    else:
        if isinstance(shap_values, list):
            mean_abs_shaps = np.abs(shap_values[0])   # (n_samples, m)
        else:
            mean_abs_shaps = np.abs(shap_values)      # (n_samples, m)

    print("mean_abs_shaps.shape =", mean_abs_shaps.shape)
    if mean_abs_shaps.ndim == 3:
        # (n_samples, m, n_classes)
        importances = mean_abs_shaps.mean(axis=(0,2)) # => (m,)
    else:
        importances = mean_abs_shaps.mean(axis=0)     # => (m,)

    print("importances.shape =", importances.shape)  # (m,)
    indices = np.argsort(importances)[::-1]

    m = pca_.n_components_
    pc_names = [f"PC{i+1}" for i in range(m)]
    shap_sorted_components = [pc_names[i] for i in indices]
    shap_sorted_importances = importances[indices]

    print("\n=== Ranking PCA components by SHAP ===")
    for i, comp in enumerate(shap_sorted_components):
        print(f"{i+1}. {comp} => mean|SHAP|={shap_sorted_importances[i]:.4f}")

    # ========== 4) Cross-validation pipeline ==========
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mcc_scorer = make_scorer(matthews_corrcoef)
    scoring = {"accuracy":"accuracy", "f1_macro":"f1_macro", "mcc":mcc_scorer}

    cv_scores = cross_validate(
        pca_pipeline, X_train, y_train,
        cv=cv, scoring=scoring, n_jobs=-1
    )

    acc_mean = np.mean(cv_scores["test_accuracy"])
    f1_mean  = np.mean(cv_scores["test_f1_macro"])
    mcc_mean = np.mean(cv_scores["test_mcc"])

    print(f"\nCV results (pipeline with PCA(80% var) + SVC(RBF)):")
    print(f"Accuracy={acc_mean:.3f}, F1_macro={f1_mean:.3f}, MCC={mcc_mean:.3f}")

    # ========== 5) Final Evaluation on Test Set ==========
    y_pred_test = pca_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="macro")
    test_mcc = matthews_corrcoef(y_test, y_pred_test)

    print("\n=== Final test results (no tuning) ===")
    print(f"Accuracy={test_acc:.3f}")
    print(f"F1-macro={test_f1:.3f}")
    print(f"MCC={test_mcc:.3f}")

    # ========== 6) Hyperparameter Tuning ==========
    param_grid = {
        "svc__C":     [0.01, 0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1]
    }
    grid = GridSearchCV(
        estimator=pca_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("\n=== Tuning results ===")
    print("Best params:", grid.best_params_)
    print(f"Best CV F1-macro={grid.best_score_:.3f}")

    best_model = grid.best_estimator_
    y_pred_test_tuned = best_model.predict(X_test)

    test_acc_tuned = accuracy_score(y_test, y_pred_test_tuned)
    test_f1_tuned  = f1_score(y_test, y_pred_test_tuned, average="macro")
    test_mcc_tuned = matthews_corrcoef(y_test, y_pred_test_tuned)

    print("\n=== Final test with tuned hyperparameters (PCA + SVC) ===")
    print(f"Accuracy={test_acc_tuned:.3f}")
    print(f"F1-macro={test_f1_tuned:.3f}")
    print(f"MCC={test_mcc_tuned:.3f}")

    # ========== 7) Save and Plot Metrics ==========
    
    # Create DataFrame with results
    metrics_df = pd.DataFrame({
        "Method": ["CV (No Tuning)", "Test (No Tuning)", "Test (Tuned)"],
        "Accuracy": [acc_mean, test_acc, test_acc_tuned],
        "F1_macro": [f1_mean, test_f1, test_f1_tuned],
        "MCC": [mcc_mean, test_mcc, test_mcc_tuned]
    })

    print("\n=== Summary of Metrics ===")
    print(metrics_df)

    # Convert to long format
    df_plot = metrics_df.melt(id_vars="Method", var_name="Metric", value_name="Score")

    # Bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_plot, x="Method", y="Score", hue="Metric", palette="Set2")
    plt.ylim(0, 1)  # scale 0 to 1
    plt.title("Comparison of Metrics (No Tuning vs Tuned)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def SVC_lime():
    from src.utils import data_imputation
    from src.collect_data import load_data
    # LIME
    # jeśli nie masz zainstalowane; w notebooku można
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    
    data = load_data("final_features.csv")
    data = data_imputation(data)
    # ========== 1) Wczytanie i podział danych ==========
    # Zakładamy, że masz wczytaną ramkę 'data' z kolumną "target"

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42
    )

    # ========== 2) Pipeline: Standaryzacja + SVC(probability=True) ==========

    svc_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel='rbf', probability=True, random_state=42))
    ])

    # Dopasowujemy pipeline na zbiorze treningowym
    svc_pipeline.fit(X_train, y_train)

    # ========== 3) Cross-validation bez strojenia ==========

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mcc_scorer = make_scorer(matthews_corrcoef)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "mcc": mcc_scorer}

    cv_scores = cross_validate(
        svc_pipeline, X_train, y_train, 
        cv=cv, scoring=scoring, n_jobs=-1
    )

    acc_mean = np.mean(cv_scores["test_accuracy"])
    f1_mean  = np.mean(cv_scores["test_f1_macro"])
    mcc_mean = np.mean(cv_scores["test_mcc"])

    print("\n=== CV results (SVC) ===")
    print(f"Accuracy={acc_mean:.3f}, F1_macro={f1_mean:.3f}, MCC={mcc_mean:.3f}")

    # ========== 4) Ocena na zbiorze testowym (bez tuningu) ==========

    y_pred_test = svc_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="macro")
    test_mcc = matthews_corrcoef(y_test, y_pred_test)

    print("\n=== Final test results (no tuning) ===")
    print(f"Accuracy={test_acc:.3f}")
    print(f"F1-macro={test_f1:.3f}")
    print(f"MCC={test_mcc:.3f}")

    # ========== 5) Strojenie hiperparametrów (GridSearchCV) ==========

    param_grid = {
        "svc__C":     [0.01, 0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1]
    }

    grid = GridSearchCV(
        estimator=svc_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("\n=== Tuning results ===")
    print("Best params:", grid.best_params_)
    print(f"Best CV F1-macro={grid.best_score_:.3f}")

    # Tworzymy finalny model
    best_model = grid.best_estimator_
    y_pred_test_tuned = best_model.predict(X_test)

    test_acc_tuned = accuracy_score(y_test, y_pred_test_tuned)
    test_f1_tuned  = f1_score(y_test, y_pred_test_tuned, average="macro")
    test_mcc_tuned = matthews_corrcoef(y_test, y_pred_test_tuned)

    print("\n=== Final test with tuned hyperparams (SVC) ===")
    print(f"Accuracy={test_acc_tuned:.3f}")
    print(f"F1-macro={test_f1_tuned:.3f}")
    print(f"MCC={test_mcc_tuned:.3f}")

    # ========== 6) LIME do objaśnienia lokalnego i globalnego rankingu cech ==========

    # a) Definiujemy LimeTabularExplainer na TRAIN
    explainer_lime = LimeTabularExplainer(
        training_data = X_train.values,      # numpy array
        feature_names = X_train.columns.tolist(), 
        class_names   = np.unique(y_train).astype(str).tolist(),
        discretize_continuous = True
    )

    # b) Funkcja do predykcji proba w stylu LIME (używamy best_model)
    def predict_proba_func(X_2d):
        """
        X_2d: np.array (n_samples, n_features)
        Zwraca prawdopodobieństwa klas best_model
        """
        return best_model.predict_proba(X_2d)

    # c) Lokalna interpretacja LIME => pętla dla kilkudziesięciu próbek
    import collections

    n_samples_for_global = 30  # np. 30
    X_sub_global = X_test.sample(n_samples_for_global, random_state=123)

    feature_weights = collections.defaultdict(float)
    count_feature   = collections.defaultdict(int)

    for i, (idx, row) in enumerate(X_sub_global.iterrows()):
        x_row = row.values  # 1D array
        # Wyjaśniamy lokalnie
        exp = explainer_lime.explain_instance(
            data_row = x_row,
            predict_fn = predict_proba_func,
            num_features = X_train.shape[1],  # weźmy wszystkie cechy
            top_labels = 1  # 1 klasa (ta, którą model przypisał)
        )
        label_expl = exp.available_labels()[0]
        exp_map = exp.as_list(label=label_expl)
        # exp_map: lista [(cecha, waga), ...]

        # Zbieramy wagi (wartości bezwzględne)
        for (feat_name, w) in exp_map:
            # feat_name może być np. "col_3 <= 5.2" - bo jest discretize_continuous
            # Zachowajmy klucz=feat_name lub uprośćmy 
            feature_weights[feat_name] += abs(w)
            count_feature[feat_name]   += 1

    # d) Wyliczamy średnie wagi
    global_lime_ranking = []
    for feat_name, total_w in feature_weights.items():
        avg_w = total_w / count_feature[feat_name]
        global_lime_ranking.append((feat_name, avg_w))

    # Sortujemy malejąco
    global_lime_ranking.sort(key=lambda x: x[1], reverse=True)

    # e) Wyświetlamy top 15
    print("\n=== Global LIME Ranking (avg abs weight) ===")
    for i, (fn, val) in enumerate(global_lime_ranking[:15], start=1):
        print(f"{i}. {fn} => {val:.4f}")

    # ========== 7) Podsumowanie metryk w DataFrame i wykres ==========

    metrics_df = pd.DataFrame({
        "Method": ["CV (No Tuning)", "Test (No Tuning)", "Test (Tuned)"],
        "Accuracy": [acc_mean, test_acc, test_acc_tuned],
        "F1_macro": [f1_mean, test_f1, test_f1_tuned],
        "MCC": [mcc_mean, test_mcc, test_mcc_tuned]
    })

    print("\n=== Summary of Metrics ===")
    print(metrics_df)

    df_plot = metrics_df.melt(id_vars="Method", var_name="Metric", value_name="Score")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_plot, x="Method", y="Score", hue="Metric", palette="Set2")
    plt.ylim(0, 1)
    plt.title("Comparison of Metrics: No Tuning vs Tuned (SVC)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
