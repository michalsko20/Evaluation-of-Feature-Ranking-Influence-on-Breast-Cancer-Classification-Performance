import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, 
    matthews_corrcoef, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, make_scorer
)
from src.collect_data import load_data, save_data

def rf_ranking():
    data = load_data("all_data.csv")
    # ========= 1) Data Preparation =========
    # Assume 'data' is a DataFrame with no missing values and 'target' is the class label column
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
    return importances

def mlp_classification():
    combined_df = load_data("all_data.csv")
    # --------------------------------------------------
    # 0) Przygotowanie danych
    # --------------------------------------------------
    # Zakładamy, że w zmiennej `combined_df` masz już wczytane dane
    # z kolumnami cech i kolumną "target"

    le = LabelEncoder()
    combined_df["target"] = le.fit_transform(combined_df["target"])

    X = combined_df.drop(columns=["target"])
    y = combined_df["target"].values

    # Dla pewności: kolumny jako string
    X.columns = X.columns.astype(str)

    # Standardyzacja
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Załóżmy, że masz już obliczone importance z np. RandomForest (base_model)
    importances = rf_ranking()
    feature_names = X.columns

    # Sortowanie cech wg ważności
    indices = np.argsort(importances)[::-1]
    sorted_features = feature_names[indices]
    sorted_importances = importances[indices]

    # Lista wartości k
    feature_counts = list(range(50, 751, 10))

    # --------------------------------------------------
    # 1) Prosta sieć MLP
    # --------------------------------------------------
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, output_dim=2):
            super(SimpleMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
        def forward(self, x):
            return self.net(x)

    # --------------------------------------------------
    # 2) Funkcja do trenowania modelu + metryki treningowe
    # --------------------------------------------------
    def train_model(
        model,
        optimizer,
        criterion,
        X_train_tensor,
        y_train_tensor,
        n_epochs=20,
        batch_size=32
    ):
        """
        Trenuje model przez n_epochs na zbiorze treningowym (bez walidacji).
        Zwraca historię (listy) metryk:
        - train_losses
        - train_accuracies
        """
        train_losses = []
        train_accuracies = []
        
        dataset_size = X_train_tensor.size(0)
        
        for epoch in range(n_epochs):
            model.train()
            # Shuffle indices for mini-batch
            indices = torch.randperm(dataset_size)
            
            running_loss = 0.0
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                # Forward
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backprop + opt step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * (end_idx - start_idx)
            
            # Średnia strata w tej epoce
            epoch_train_loss = running_loss / dataset_size
            
            # Obliczamy accuracy na CAŁYM zbiorze treningowym w tej epoce
            model.eval()
            with torch.no_grad():
                outputs_train = model(X_train_tensor)
                preds_train = outputs_train.argmax(dim=1).cpu().numpy()
                y_true_train = y_train_tensor.cpu().numpy()
                train_acc = accuracy_score(y_true_train, preds_train)
            
            train_losses.append(epoch_train_loss)
            train_accuracies.append(train_acc)
            
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        return train_losses, train_accuracies

    # --------------------------------------------------
    # 3) Funkcja do obliczania metryk końcowych (na walidacji)
    # --------------------------------------------------
    def compute_metrics(model, X_tensor, y_tensor):
        """
        Zwraca (loss, accuracy, f1, mcc) dla zadanych danych (walidacyjnych lub testowych),
        przy wykorzystaniu cross-entropy jako funkcji straty.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true = y_tensor.cpu().numpy()
            
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, average='macro')
            mcc = matthews_corrcoef(y_true, preds)
        return loss.item(), acc, f1, mcc

    # --------------------------------------------------
    # 4) Funkcja do rysowania średnich metryk treningowych
    # --------------------------------------------------
    def plot_avg_train_metrics(all_fold_train_losses, all_fold_train_accuracies, n_epochs):
        """
        Rysuje wykresy średnich strat i dokładności treningowych (po foldach).
        """
        # Uśrednianie strat i dokładności po foldach (dla każdej epoki)
        avg_train_losses = np.mean(all_fold_train_losses, axis=0)
        avg_train_accuracies = np.mean(all_fold_train_accuracies, axis=0)
        
        epochs = range(1, n_epochs + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Wykres strat (loss)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, avg_train_losses, label='Average Train Loss', marker='o')
        plt.title('Average Training Loss Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Wykres dokładności (accuracy)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, avg_train_accuracies, label='Average Train Accuracy', marker='o')
        plt.title('Average Training Accuracy Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # 5) Cross-Validation z agregacją metryk
    # --------------------------------------------------
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Listy na metryki treningowe z każdego folda
    all_fold_train_losses = []
    all_fold_train_accuracies = []

    n_epochs = 50  # Liczba epok

    for k in feature_counts:
        # Wybierz top-k cech
        top_k_features = sorted_features[:k]
        
        # Wyciągnij te kolumny ze znormalizowanego X
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        X_k = X_scaled_df[top_k_features].values
        
        # Listy na metryki końcowe z CV
        fold_accuracies = []
        fold_f1s = []
        fold_mccs = []
        
        # Cross Validation
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_k, y)):
            X_train_fold, X_val_fold = X_k[train_idx], X_k[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Tworzymy tensory
            X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long)
            X_val_tensor   = torch.tensor(X_val_fold,   dtype=torch.float32)
            y_val_tensor   = torch.tensor(y_val_fold,   dtype=torch.long)
            
            # Definiujemy model
            output_dim = len(np.unique(y))
            model = SimpleMLP(input_dim=k, hidden_dim=64, output_dim=output_dim)
            
            # Optymalizator i loss
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Trenujemy model i zapisujemy metryki treningowe
            train_losses, train_accuracies = train_model(
                model,
                optimizer,
                criterion,
                X_train_tensor,
                y_train_tensor,
                n_epochs=n_epochs,
                batch_size=32
            )
            
            # Dodajemy metryki treningowe do list foldów
            all_fold_train_losses.append(train_losses)
            all_fold_train_accuracies.append(train_accuracies)
            
            # Ewaluacja na zbiorze walidacyjnym
            _, acc, f1, mcc = compute_metrics(model, X_val_tensor, y_val_tensor)
            fold_accuracies.append(acc)
            fold_f1s.append(f1)
            fold_mccs.append(mcc)
        
        # Średnie metryki w CV
        acc_mean = np.mean(fold_accuracies)
        f1_mean = np.mean(fold_f1s)
        mcc_mean = np.mean(fold_mccs)
        
        results.append({
            'num_features': k,
            'cv_accuracy': acc_mean,
            'cv_f1_macro': f1_mean,
            'cv_mcc': mcc_mean
        })

    # Po zakończeniu cross-validation, uśrednij i wyświetl metryki treningowe
    plot_avg_train_metrics(all_fold_train_losses, all_fold_train_accuracies, n_epochs)

    # Po zakończeniu pętli po k, mamy w "results" wyniki CV
    results_df = pd.DataFrame(results)
    print("\nCross-validation results (averages) for different feature counts:")
    print(results_df)

    # --------------------------------------------------
    # 6) Wybór najlepszego k (na podstawie F1-macro)
    # --------------------------------------------------
    best_k = results_df.loc[results_df['cv_f1_macro'].idxmax(), 'num_features']
    print(f"\nBest k (based on F1-macro in CV): {best_k}")

    # --------------------------------------------------
    # 7) Wykres zmian metryk końcowych vs. liczba cech
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['num_features'], results_df['cv_accuracy'], marker='o', label='CV Accuracy')
    plt.plot(results_df['num_features'], results_df['cv_f1_macro'], marker='s', label='CV F1 Macro')
    plt.plot(results_df['num_features'], results_df['cv_mcc'], marker='^', label='CV MCC')
    plt.xlabel('Number of Selected Features (k)')
    plt.ylabel('Metrics (Average from 5-fold CV)')
    plt.title('Cross-Validation Results (PyTorch MLP) vs. Number of Features')
    plt.grid(True)
    plt.legend()
    plt.show()

def mlp_classification_upgraded():
    combined_df = load_data("all_data.csv")

    le = LabelEncoder()
    combined_df["target"] = le.fit_transform(combined_df["target"])

    X = combined_df.drop(columns=["target"])
    y = combined_df["target"].values

    # Dla pewności: kolumny jako string
    X.columns = X.columns.astype(str)

    # Standardyzacja
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Załóżmy, że masz już obliczone importance z np. RandomForest (base_model)
    importances = rf_ranking()
    feature_names = X.columns

    # Sortowanie cech wg ważności
    indices = np.argsort(importances)[::-1]
    sorted_features = feature_names[indices]
    sorted_importances = importances[indices]

    # Lista wartości k
    feature_counts = list(range(50, 751, 10))

    # --------------------------------------------------
    # 1) Prosta sieć MLP
    # --------------------------------------------------
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, output_dim=2):
            super(SimpleMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
        def forward(self, x):
            return self.net(x)

    # --------------------------------------------------
    # 2) Funkcja do trenowania modelu + metryki treningowe
    # --------------------------------------------------
    def train_model(
        model,
        optimizer,
        criterion,
        X_train_tensor,
        y_train_tensor,
        n_epochs=20,
        batch_size=32
    ):
        """
        Trenuje model przez n_epochs na zbiorze treningowym (bez walidacji).
        Zwraca historię (listy) metryk:
        - train_losses
        - train_accuracies
        """
        train_losses = []
        train_accuracies = []
        
        dataset_size = X_train_tensor.size(0)
        
        for epoch in range(n_epochs):
            model.train()
            # Shuffle indices for mini-batch
            indices = torch.randperm(dataset_size)
            
            running_loss = 0.0
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                # Forward
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backprop + opt step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * (end_idx - start_idx)
            
            # Średnia strata w tej epoce
            epoch_train_loss = running_loss / dataset_size
            
            # Obliczamy accuracy na CAŁYM zbiorze treningowym w tej epoce
            model.eval()
            with torch.no_grad():
                outputs_train = model(X_train_tensor)
                preds_train = outputs_train.argmax(dim=1).cpu().numpy()
                y_true_train = y_train_tensor.cpu().numpy()
                train_acc = accuracy_score(y_true_train, preds_train)
            
            train_losses.append(epoch_train_loss)
            train_accuracies.append(train_acc)
            
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        return train_losses, train_accuracies

    # --------------------------------------------------
    # 3) Funkcja do obliczania metryk końcowych (na walidacji)
    # --------------------------------------------------
    def compute_metrics(model, X_tensor, y_tensor):
        """
        Zwraca (loss, accuracy, f1, mcc) dla zadanych danych (walidacyjnych lub testowych),
        przy wykorzystaniu cross-entropy jako funkcji straty.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true = y_tensor.cpu().numpy()
            
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, average='macro')
            mcc = matthews_corrcoef(y_true, preds)
        return loss.item(), acc, f1, mcc

    # --------------------------------------------------
    # 4) Funkcja do rysowania średnich metryk treningowych
    # --------------------------------------------------
    def plot_avg_train_metrics(all_fold_train_losses, all_fold_train_accuracies, n_epochs):
        """
        Rysuje wykresy średnich strat i dokładności treningowych (po foldach).
        """
        # Uśrednianie strat i dokładności po foldach (dla każdej epoki)
        avg_train_losses = np.mean(all_fold_train_losses, axis=0)
        avg_train_accuracies = np.mean(all_fold_train_accuracies, axis=0)
        
        epochs = range(1, n_epochs + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Wykres strat (loss)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, avg_train_losses, label='Average Train Loss', marker='o')
        plt.title('Average Training Loss Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Wykres dokładności (accuracy)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, avg_train_accuracies, label='Average Train Accuracy', marker='o')
        plt.title('Average Training Accuracy Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # 5) Cross-Validation z agregacją metryk
    # --------------------------------------------------
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Listy na metryki treningowe z każdego folda
    all_fold_train_losses = []
    all_fold_train_accuracies = []

    n_epochs = 50  # Liczba epok

    for k in feature_counts:
        # Wybierz top-k cech
        top_k_features = sorted_features[:k]
        
        # Wyciągnij te kolumny ze znormalizowanego X
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        X_k = X_scaled_df[top_k_features].values
        
        # Listy na metryki końcowe z CV
        fold_accuracies = []
        fold_f1s = []
        fold_mccs = []
        
        # Cross Validation
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_k, y)):
            X_train_fold, X_val_fold = X_k[train_idx], X_k[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Tworzymy tensory
            X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long)
            X_val_tensor   = torch.tensor(X_val_fold,   dtype=torch.float32)
            y_val_tensor   = torch.tensor(y_val_fold,   dtype=torch.long)
            
            # Definiujemy model
            output_dim = len(np.unique(y))
            model = SimpleMLP(input_dim=k, hidden_dim=64, output_dim=output_dim)
            
            # Optymalizator i loss
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Trenujemy model i zapisujemy metryki treningowe
            train_losses, train_accuracies = train_model(
                model,
                optimizer,
                criterion,
                X_train_tensor,
                y_train_tensor,
                n_epochs=n_epochs,
                batch_size=32
            )
            
            # Dodajemy metryki treningowe do list foldów
            all_fold_train_losses.append(train_losses)
            all_fold_train_accuracies.append(train_accuracies)
            
            # Ewaluacja na zbiorze walidacyjnym
            _, acc, f1, mcc = compute_metrics(model, X_val_tensor, y_val_tensor)
            fold_accuracies.append(acc)
            fold_f1s.append(f1)
            fold_mccs.append(mcc)
        
        # Średnie metryki w CV
        acc_mean = np.mean(fold_accuracies)
        f1_mean = np.mean(fold_f1s)
        mcc_mean = np.mean(fold_mccs)
        
        results.append({
            'num_features': k,
            'cv_accuracy': acc_mean,
            'cv_f1_macro': f1_mean,
            'cv_mcc': mcc_mean
        })

    # Po zakończeniu cross-validation, uśrednij i wyświetl metryki treningowe
    plot_avg_train_metrics(all_fold_train_losses, all_fold_train_accuracies, n_epochs)

    # Po zakończeniu pętli po k, mamy w "results" wyniki CV
    results_df = pd.DataFrame(results)
    print("\nCross-validation results (averages) for different feature counts:")
    print(results_df)

    # --------------------------------------------------
    # 6) Wybór najlepszego k (na podstawie F1-macro)
    # --------------------------------------------------
    best_k = results_df.loc[results_df['cv_f1_macro'].idxmax(), 'num_features']
    print(f"\nBest k (based on F1-macro in CV): {best_k}")

    # --------------------------------------------------
    # 7) Wykres zmian metryk końcowych vs. liczba cech
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['num_features'], results_df['cv_accuracy'], marker='o', label='CV Accuracy')
    plt.plot(results_df['num_features'], results_df['cv_f1_macro'], marker='s', label='CV F1 Macro')
    plt.plot(results_df['num_features'], results_df['cv_mcc'], marker='^', label='CV MCC')
    plt.xlabel('Number of Selected Features (k)')
    plt.ylabel('Metrics (Average from 5-fold CV)')
    plt.title('Cross-Validation Results (PyTorch MLP) vs. Number of Features')
    plt.grid(True)
    plt.legend()
    plt.show()