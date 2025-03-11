import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, make_scorer
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class SVCFeatureRankings:
    def __init__(self, data_path="final_features.csv", test_size=0.2, random_state=42):
        from src.utils import data_imputation
        from src.collect_data import load_data

        data = load_data(data_path)
        self.data = data_imputation(data)
        self.X = self.data.drop(columns=["target"])
        self.y = self.data["target"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=random_state,
        )

    # --- Metody rankingów ---
    def get_ranking_filter_mi(self, X_tr, y_tr):
        mi_values = mutual_info_classif(
            X_tr, y_tr, discrete_features=False, random_state=42
        )
        mi_indices = np.argsort(mi_values)[::-1]
        return X_tr.columns[mi_indices]

    def get_ranking_svc_coef(self, X_tr, y_tr):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="linear", max_iter=20000, random_state=42)),
            ]
        )
        pipe.fit(X_tr, y_tr)
        svc_linear = pipe.named_steps["svc"]
        importances = np.abs(svc_linear.coef_).mean(axis=0)
        indices = np.argsort(importances)[::-1]
        return X_tr.columns[indices]

    def get_ranking_rfe(self, X_tr, y_tr):
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        svc_lin = SVC(kernel="linear", max_iter=20000, random_state=42)
        rfe = RFE(estimator=svc_lin, n_features_to_select=1, step=1)
        rfe.fit(X_tr_scaled, y_tr)
        indices = np.argsort(rfe.ranking_)
        return X_tr.columns[indices]

    def get_ranking_permutation(self, X_tr, y_tr):
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", random_state=42))]
        )
        pipe.fit(X_tr, y_tr)
        X_tr_scaled = pipe.named_steps["scaler"].transform(X_tr)
        svc_rbf = pipe.named_steps["svc"]
        perm = permutation_importance(
            svc_rbf, X_tr_scaled, y_tr, scoring="accuracy", n_repeats=5, random_state=42
        )
        importances = perm.importances_mean
        indices = np.argsort(importances)[::-1]
        return X_tr.columns[indices]

    # --- Walidacja i ocena ---
    def evaluate_rankings(self, feature_counts=[x for x in range(5, 200, 10)]):
        ranking_methods = {
            "Filter_MI": self.get_ranking_filter_mi,
            "SVC_coef": self.get_ranking_svc_coef,
            "RFE": self.get_ranking_rfe,
            "PermImp": self.get_ranking_permutation,
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        mcc_scorer = make_scorer(matthews_corrcoef)
        scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "mcc": mcc_scorer}

        all_results = []

        for rank_name, rank_func in ranking_methods.items():
            print(f"\n=== RANKING: {rank_name} ===")
            sorted_feats = rank_func(self.X_train, self.y_train)
            results = []

            for k in feature_counts:
                top_k = sorted_feats[:k]
                model_k = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("svc", SVC(kernel="rbf", random_state=42)),
                    ]
                )
                cv_scores = cross_validate(
                    model_k,
                    self.X_train[top_k],
                    self.y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                )
                acc_mean = np.mean(cv_scores["test_accuracy"])
                f1_mean = np.mean(cv_scores["test_f1_macro"])
                mcc_mean = np.mean(cv_scores["test_mcc"])

                results.append(
                    {
                        "ranking": rank_name,
                        "num_features": k,
                        "cv_accuracy": acc_mean,
                        "cv_f1_macro": f1_mean,
                        "cv_mcc": mcc_mean,
                    }
                )
            df_r = pd.DataFrame(results)
            print(df_r)

            best_idx = df_r["cv_f1_macro"].idxmax()
            best_k = df_r.loc[best_idx, "num_features"]
            print(f"Best k for {rank_name}: {best_k}")

            # Ewaluacja na zbiorze testowym z najlepszą liczbą cech
            final_feats = sorted_feats[:best_k]
            final_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="rbf", random_state=42)),
                ]
            )
            final_model.fit(self.X_train[final_feats], self.y_train)
            y_pred_test = final_model.predict(self.X_test[final_feats])
            test_acc = accuracy_score(self.y_test, y_pred_test)
            test_f1 = f1_score(self.y_test, y_pred_test, average="macro")
            test_mcc = matthews_corrcoef(self.y_test, y_pred_test)
            print(
                f"[{rank_name}] Test results for k={best_k} => "
                f"ACC={test_acc:.3f}, F1={test_f1:.3f}, MCC={test_mcc:.3f}"
            )

            all_results.extend(results)

        self.all_results_df = pd.DataFrame(all_results)
        return self.all_results_df

    def hyperparameter_tuning(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                "svc__C": [0.01, 0.1, 1, 10, 100],
                "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1],
            }
        tuned_results = []
        ranking_methods = {
            "Filter_MI": self.get_ranking_filter_mi,
            "SVC_coef": self.get_ranking_svc_coef,
            "RFE": self.get_ranking_rfe,
            "PermImp": self.get_ranking_permutation,
        }

        for rank_name, rank_func in ranking_methods.items():
            best_k = self.all_results_df[self.all_results_df["ranking"] == rank_name][
                "num_features"
            ].max()
            sorted_feats = rank_func(self.X_train, self.y_train)
            final_feats = sorted_feats[:best_k]
            print(f"\n>>> Tuning hyperparams for ranking={rank_name}, k={best_k}")

            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="rbf", random_state=42)),
                ]
            )
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=5,
                scoring="f1_macro",
                n_jobs=-1,
            )
            grid.fit(self.X_train[final_feats], self.y_train)
            print(f"Best params for {rank_name}: {grid.best_params_}")
            print(f"Best CV score (f1_macro) for {rank_name}: {grid.best_score_:.4f}")
            best_model = grid.best_estimator_
            y_pred_test = best_model.predict(self.X_test[final_feats])
            test_acc = accuracy_score(self.y_test, y_pred_test)
            test_f1 = f1_score(self.y_test, y_pred_test, average="macro")
            test_mcc = matthews_corrcoef(self.y_test, y_pred_test)
            tuned_results.append(
                {
                    "ranking": rank_name,
                    "best_k": best_k,
                    "best_params": grid.best_params_,
                    "test_acc_tuned": test_acc,
                    "test_f1_macro_tuned": test_f1,
                    "test_mcc_tuned": test_mcc,
                }
            )
        tuned_df = pd.DataFrame(tuned_results)
        print("\n=== Tuned results for each ranking ===")
        print(tuned_df)
        return tuned_df

    def run(self):
        results_df = self.evaluate_rankings()
        tuned_df = self.hyperparameter_tuning()
        return results_df, tuned_df


class PCA_SHAP_SVC_Ranker:
    def __init__(
        self,
        data_path="final_features.csv",
        test_size=0.2,
        random_state=42,
        explained_variance=0.80,
    ):
        from src.utils import data_imputation
        from src.collect_data import load_data

        data = load_data(data_path)
        self.data = data_imputation(data)

        self.X = self.data.drop(columns=["target"])
        self.y = self.data["target"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=random_state,
        )

        self.explained_variance = explained_variance

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=self.explained_variance)),
                ("svc", SVC(kernel="rbf", probability=True, random_state=random_state)),
            ]
        )

    def fit_pipeline(self):
        self.pipeline.fit(self.X_train, self.y_train)
        self.scaler_ = self.pipeline.named_steps["scaler"]
        self.pca_ = self.pipeline.named_steps["pca"]
        self.svc_ = self.pipeline.named_steps["svc"]
        print("Number of components after PCA =", self.pca_.n_components_)

    def compute_shap_ranking(self, nsamples=100, l1_reg="num_features(10)"):
        import shap

        n_bg = min(100, len(self.X_train))
        n_subset = min(200, len(self.X_train))
        X_bg = self.X_train.sample(n_bg, random_state=42)
        X_sub = self.X_train.sample(n_subset, random_state=123)

        # Skalowanie i PCA
        X_bg_pca = self.pca_.transform(self.scaler_.transform(X_bg))
        X_sub_pca = self.pca_.transform(self.scaler_.transform(X_sub))

        def custom_predict_proba_pca_only(X_):
            return self.svc_.predict_proba(X_)

        explainer = shap.KernelExplainer(
            model=custom_predict_proba_pca_only, data=X_bg_pca, link="logit"
        )
        shap_values = explainer.shap_values(X_sub_pca, nsamples=nsamples, l1_reg=l1_reg)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            abs_shaps = [np.abs(sv) for sv in shap_values]
            mean_abs_shaps = np.mean(abs_shaps, axis=0)
        else:
            if isinstance(shap_values, list):
                mean_abs_shaps = np.abs(shap_values[0])
            else:
                mean_abs_shaps = np.abs(shap_values)

        if mean_abs_shaps.ndim == 3:
            importances = mean_abs_shaps.mean(axis=(0, 2))
        else:
            importances = mean_abs_shaps.mean(axis=0)

        indices = np.argsort(importances)[::-1]
        m = self.pca_.n_components_
        pc_names = [f"PC{i+1}" for i in range(m)]
        self.shap_sorted_components = [pc_names[i] for i in indices]
        self.shap_sorted_importances = importances[indices]
        print("\n=== Ranking PCA components by SHAP ===")
        for i, comp in enumerate(self.shap_sorted_components):
            print(f"{i+1}. {comp} => mean|SHAP|={self.shap_sorted_importances[i]:.4f}")

    def cross_validate_pipeline(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        mcc_scorer = make_scorer(matthews_corrcoef)
        scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "mcc": mcc_scorer}
        cv_scores = cross_validate(
            self.pipeline, self.X_train, self.y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        self.cv_results = {
            "accuracy": np.mean(cv_scores["test_accuracy"]),
            "f1_macro": np.mean(cv_scores["test_f1_macro"]),
            "mcc": np.mean(cv_scores["test_mcc"]),
        }
        print(
            f"\nCV results (pipeline with PCA({self.explained_variance*100:.0f}% var) + SVC(RBF)):"
        )
        print(
            f"Accuracy={self.cv_results['accuracy']:.3f}, F1_macro={self.cv_results['f1_macro']:.3f}, MCC={self.cv_results['mcc']:.3f}"
        )

    def evaluate_test(self):
        y_pred_test = self.pipeline.predict(self.X_test)
        self.test_results = {
            "accuracy": accuracy_score(self.y_test, y_pred_test),
            "f1_macro": f1_score(self.y_test, y_pred_test, average="macro"),
            "mcc": matthews_corrcoef(self.y_test, y_pred_test),
        }
        print("\n=== Final test results (no tuning) ===")
        print(f"Accuracy={self.test_results['accuracy']:.3f}")
        print(f"F1-macro={self.test_results['f1_macro']:.3f}")
        print(f"MCC={self.test_results['mcc']:.3f}")

    def hyperparameter_tuning(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                "svc__C": [0.01, 0.1, 1, 10, 100],
                "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1],
            }
        grid = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid.fit(self.X_train, self.y_train)
        print("\n=== Tuning results ===")
        print("Best params:", grid.best_params_)
        print(f"Best CV F1-macro={grid.best_score_:.3f}")
        self.best_model = grid.best_estimator_
        y_pred_test_tuned = self.best_model.predict(self.X_test)
        self.test_results_tuned = {
            "accuracy": accuracy_score(self.y_test, y_pred_test_tuned),
            "f1_macro": f1_score(self.y_test, y_pred_test_tuned, average="macro"),
            "mcc": matthews_corrcoef(self.y_test, y_pred_test_tuned),
        }
        print("\n=== Final test with tuned hyperparameters (PCA + SVC) ===")
        print(f"Accuracy={self.test_results_tuned['accuracy']:.3f}")
        print(f"F1-macro={self.test_results_tuned['f1_macro']:.3f}")
        print(f"MCC={self.test_results_tuned['mcc']:.3f}")

    def plot_metrics(self):
        metrics_df = pd.DataFrame(
            {
                "Method": ["CV (No Tuning)", "Test (No Tuning)", "Test (Tuned)"],
                "Accuracy": [
                    self.cv_results["accuracy"],
                    self.test_results["accuracy"],
                    self.test_results_tuned["accuracy"],
                ],
                "F1_macro": [
                    self.cv_results["f1_macro"],
                    self.test_results["f1_macro"],
                    self.test_results_tuned["f1_macro"],
                ],
                "MCC": [
                    self.cv_results["mcc"],
                    self.test_results["mcc"],
                    self.test_results_tuned["mcc"],
                ],
            }
        )
        print("\n=== Summary of Metrics ===")
        print(metrics_df)
        df_plot = metrics_df.melt(
            id_vars="Method", var_name="Metric", value_name="Score"
        )
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_plot, x="Method", y="Score", hue="Metric", palette="Set2")
        plt.ylim(0, 1)
        plt.title("Comparison of Metrics (No Tuning vs Tuned)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    def run(self):
        self.fit_pipeline()
        self.compute_shap_ranking()
        self.cross_validate_pipeline()
        self.evaluate_test()
        self.hyperparameter_tuning()
        self.plot_metrics()


class SVC_Lime_Analyzer:
    def __init__(self, data_path="final_features.csv", test_size=0.2, random_state=42):
        from src.utils import data_imputation
        from src.collect_data import load_data

        data = load_data(data_path)
        self.data = data_imputation(data)

        self.X = self.data.drop(columns=["target"])
        self.y = self.data["target"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=random_state,
        )

    def train_pipeline(self):
        self.svc_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        )
        self.svc_pipeline.fit(self.X_train, self.y_train)

    def evaluate_cv(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        mcc_scorer = make_scorer(matthews_corrcoef)
        scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "mcc": mcc_scorer}
        cv_scores = cross_validate(
            self.svc_pipeline,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        self.cv_results = {
            "accuracy": np.mean(cv_scores["test_accuracy"]),
            "f1_macro": np.mean(cv_scores["test_f1_macro"]),
            "mcc": np.mean(cv_scores["test_mcc"]),
        }
        print("\n=== CV results (SVC) ===")
        print(
            f"Accuracy={self.cv_results['accuracy']:.3f}, F1_macro={self.cv_results['f1_macro']:.3f}, MCC={self.cv_results['mcc']:.3f}"
        )

    def evaluate_test(self):
        y_pred_test = self.svc_pipeline.predict(self.X_test)
        self.test_results = {
            "accuracy": accuracy_score(self.y_test, y_pred_test),
            "f1_macro": f1_score(self.y_test, y_pred_test, average="macro"),
            "mcc": matthews_corrcoef(self.y_test, y_pred_test),
        }
        print("\n=== Final test results (no tuning) ===")
        print(f"Accuracy={self.test_results['accuracy']:.3f}")
        print(f"F1-macro={self.test_results['f1_macro']:.3f}")
        print(f"MCC={self.test_results['mcc']:.3f}")

    def hyperparameter_tuning(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                "svc__C": [0.01, 0.1, 1, 10, 100],
                "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1],
            }
        grid = GridSearchCV(
            estimator=self.svc_pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid.fit(self.X_train, self.y_train)
        print("\n=== Tuning results ===")
        print("Best params:", grid.best_params_)
        print(f"Best CV F1-macro={grid.best_score_:.3f}")
        self.best_model = grid.best_estimator_
        y_pred_test_tuned = self.best_model.predict(self.X_test)
        self.test_results_tuned = {
            "accuracy": accuracy_score(self.y_test, y_pred_test_tuned),
            "f1_macro": f1_score(self.y_test, y_pred_test_tuned, average="macro"),
            "mcc": matthews_corrcoef(self.y_test, y_pred_test_tuned),
        }
        print("\n=== Final test with tuned hyperparams (SVC) ===")
        print(f"Accuracy={self.test_results_tuned['accuracy']:.3f}")
        print(f"F1-macro={self.test_results_tuned['f1_macro']:.3f}")
        print(f"MCC={self.test_results_tuned['mcc']:.3f}")

    def lime_analysis(self, n_samples_for_global=30):
        import lime
        from lime.lime_tabular import LimeTabularExplainer
        import collections

        explainer_lime = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=np.unique(self.y_train).astype(str).tolist(),
            discretize_continuous=True,
        )

        def predict_proba_func(X_2d):
            return self.best_model.predict_proba(X_2d)

        feature_weights = collections.defaultdict(float)
        count_feature = collections.defaultdict(int)

        X_sub_global = self.X_test.sample(n_samples_for_global, random_state=123)
        for i, (idx, row) in enumerate(X_sub_global.iterrows()):
            x_row = row.values
            exp = explainer_lime.explain_instance(
                data_row=x_row,
                predict_fn=predict_proba_func,
                num_features=self.X_train.shape[1],
                top_labels=1,
            )
            label_expl = exp.available_labels()[0]
            exp_map = exp.as_list(label=label_expl)
            for feat_name, w in exp_map:
                feature_weights[feat_name] += abs(w)
                count_feature[feat_name] += 1
        self.global_lime_ranking = []
        for feat_name, total_w in feature_weights.items():
            avg_w = total_w / count_feature[feat_name]
            self.global_lime_ranking.append((feat_name, avg_w))
        self.global_lime_ranking.sort(key=lambda x: x[1], reverse=True)
        print("\n=== Global LIME Ranking (avg abs weight) ===")
        for i, (fn, val) in enumerate(self.global_lime_ranking[:15], start=1):
            print(f"{i}. {fn} => {val:.4f}")

    def plot_metrics(self):
        metrics_df = pd.DataFrame(
            {
                "Method": ["CV (No Tuning)", "Test (No Tuning)", "Test (Tuned)"],
                "Accuracy": [
                    self.cv_results["accuracy"],
                    self.test_results["accuracy"],
                    self.test_results_tuned["accuracy"],
                ],
                "F1_macro": [
                    self.cv_results["f1_macro"],
                    self.test_results["f1_macro"],
                    self.test_results_tuned["f1_macro"],
                ],
                "MCC": [
                    self.cv_results["mcc"],
                    self.test_results["mcc"],
                    self.test_results_tuned["mcc"],
                ],
            }
        )
        print("\n=== Summary of Metrics ===")
        print(metrics_df)
        df_plot = metrics_df.melt(
            id_vars="Method", var_name="Metric", value_name="Score"
        )
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_plot, x="Method", y="Score", hue="Metric", palette="Set2")
        plt.ylim(0, 1)
        plt.title("Comparison of Metrics: No Tuning vs Tuned (SVC)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    def run(self):
        self.train_pipeline()
        self.evaluate_cv()
        self.evaluate_test()
        self.hyperparameter_tuning()
        self.lime_analysis()
        self.plot_metrics()
