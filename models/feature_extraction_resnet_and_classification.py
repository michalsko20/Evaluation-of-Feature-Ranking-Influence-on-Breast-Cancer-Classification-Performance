import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, matthews_corrcoef,
    roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay,
    make_scorer
)
from sklearn.preprocessing import label_binarize

# -----------------------------------------------------
# 1. PATHS / CONFIGURATION
# -----------------------------------------------------
base_dir = r"D:\studia\magisterka\out"
subdirs = ["G1_larger", "G2_larger", "G3_larger"]  # subdirectories to search

label_map = {
    "G1_larger": 0,
    "G2_larger": 1,
    "G3_larger": 2
}

# Transformations for ResNet (e.g., ResNet18)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# -----------------------------------------------------
# 2. LOADING MODEL (RESNET18) - Removing the last layer
# -----------------------------------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Identity()  # Replace FC with identity: output will be a feature vector (512-d)

model.eval()  # evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -----------------------------------------------------
# 3. RECURSIVE SEARCH FOR _pre_segmented.png AND FEATURE EXTRACTION
# -----------------------------------------------------
features_list = []
labels_list = []

for subdir in subdirs:
    subdir_path = os.path.join(base_dir, subdir)
    
    for root, dirs, files in os.walk(subdir_path):
        for fname in files:
            if fname.endswith("_pre_segmented.png"):
                file_path = os.path.join(root, fname)
                
                pil_img = Image.open(file_path).convert("RGB")
                input_tensor = image_transform(pil_img)
                input_tensor = input_tensor.unsqueeze(0).to(device)  # (1,3,224,224)
                
                with torch.no_grad():
                    output = model(input_tensor)  # [1, 512] for ResNet18
                feat = output.squeeze().cpu().numpy()  # Convert to NumPy
                
                features_list.append(feat)
                labels_list.append(label_map[subdir])

features_np = np.array(features_list)  # (N, 512)
labels_np = np.array(labels_list)

print("Collected features:", features_np.shape)
print("Collected labels:", labels_np.shape)

# -----------------------------------------------------
# 4. CREATING DATAFRAME / OPTIONAL CSV SAVE
# -----------------------------------------------------
df_features = pd.DataFrame(features_np)
df_features["label"] = labels_np

df_features.to_csv("features_resnet18.csv", index=False)
print("Features saved to file: features_resnet18.csv")

# -----------------------------------------------------
# 5. TRAIN/TEST SPLIT
# -----------------------------------------------------
X = df_features.drop(columns=["label"])
y = df_features["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------
# 5.1 CROSS-VALIDATION ON TRAINING SET
# -----------------------------------------------------
clf_cv = RandomForestClassifier(n_estimators=100, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro',
    'f1_macro': 'f1_macro',
    'mcc': make_scorer(matthews_corrcoef)
}

cv_results = cross_validate(clf_cv, X_train, y_train,
                            cv=cv, scoring=scoring,
                            return_train_score=False)

cv_accuracy_mean = cv_results['test_accuracy'].mean()
cv_precision_macro_mean = cv_results['test_precision_macro'].mean()
cv_recall_macro_mean = cv_results['test_recall_macro'].mean()
cv_f1_macro_mean = cv_results['test_f1_macro'].mean()
cv_mcc_mean = cv_results['test_mcc'].mean()

print("\n=== CROSS-VALIDATION RESULTS (5-FOLD) ON TRAIN ===")
print(f"CV Accuracy (mean):         {cv_accuracy_mean:.3f}")
print(f"CV Precision Macro (mean):  {cv_precision_macro_mean:.3f}")
print(f"CV Recall Macro (mean):     {cv_recall_macro_mean:.3f}")
print(f"CV F1 Macro (mean):         {cv_f1_macro_mean:.3f}")
print(f"CV MCC (mean):              {cv_mcc_mean:.3f}")

# -----------------------------------------------------
# 5.2 FINAL TRAINING ON FULL TRAIN SET AND TESTING
# -----------------------------------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------------------------------
# 6. EVALUATION ON TEST SET
# -----------------------------------------------------
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
mcc = matthews_corrcoef(y_test, y_pred)

print("\n=== TEST SET METRICS ===")
print(f"Accuracy:         {accuracy:.3f}")
print(f"Precision(macro): {precision_macro:.3f}")
print(f"Recall(macro):    {recall_macro:.3f}")
print(f"F1(macro):        {f1_macro:.3f}")
print(f"MCC:              {mcc:.3f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=3))

# -----------------------------------------------------
# 7. CONFUSION MATRIX
# -----------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(cm, display_labels=["G1", "G2", "G3"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (RF on CNN features) - TEST")
plt.show()

# -----------------------------------------------------
# 8. MULTICLASS ROC AUC AND PLOT
# -----------------------------------------------------
y_proba = clf.predict_proba(X_test)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

plt.figure(figsize=(8, 6))
for i, class_name in enumerate(["G1", "G2", "G3"]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=class_name).plot(ax=plt.gca())

plt.title("ROC Curves (One-vs-Rest) - CNN Features + RF (TEST)")
plt.show()

roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
print(f"ROC AUC (macro, OvR, TEST): {roc_auc_macro:.3f}")

# -----------------------------------------------------
# 9. METRICS COMPARISON BAR PLOT
# -----------------------------------------------------
metrics_names = ["Precision(macro)", "Recall(macro)", "F1(macro)"]
metrics_values = [precision_macro, recall_macro, f1_macro]

plt.figure(figsize=(6, 4))
plt.bar(metrics_names, metrics_values, color=["orange", "green", "blue"])
plt.ylim([0, 1])
plt.title("Metrics Comparison (RF on ResNet18 Features) - TEST")
for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
plt.show()
