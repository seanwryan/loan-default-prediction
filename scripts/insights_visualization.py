# insights_visualization.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load the engineered dataset and feature importance
data_path = './data/hmeq_engineered.csv'
feature_importance_path = './data/feature_importance.csv'
df = pd.read_csv(data_path)
feature_importance = pd.read_csv(feature_importance_path)

# Load evaluation metrics (manually retrieved from previous scripts)
roc_auc_logistic = 0.76  # Logistic Regression AUC
roc_auc_lgbm = 0.96      # LightGBM AUC

# Consolidate all plots into fewer files

# 1. Feature Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature', palette='viridis')
plt.title("Top 10 Features Contributing to Loan Defaults (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig('./data/Consolidated_Feature_Importance.png')
plt.show()

# 2. Model Performance Comparison: Confusion Matrix + ROC Curves
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Confusion Matrix (LightGBM)
conf_matrix = np.array([[922, 32], [65, 173]])  # Replace with actual confusion matrix values
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Non-Default', 'Default'])
disp.plot(cmap='Blues', ax=ax[0], colorbar=False)
ax[0].set_title("Confusion Matrix (LightGBM)")

# ROC Curves
roc_curves = ['./plots/models/ROC_Curve.png', './plots/models/ROC_Curve_Lgbm.png']
titles = ["Logistic Regression ROC Curve", "LightGBM ROC Curve"]

for i, roc_curve_path in enumerate(roc_curves):
    img = plt.imread(roc_curve_path)
    ax[i + 1].imshow(img)
    ax[i + 1].axis('off')
    ax[i + 1].set_title(titles[i])

plt.tight_layout()
plt.savefig('./data/Model_Performance_Comparison.png')
plt.show()

# 3. Borrower Characteristics by Default Status
key_features = ['DEBTINC', 'CLAGE', 'Total_Derog_Delinquent', 'LOAN']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, feature in enumerate(key_features):
    sns.histplot(data=df, x=feature, hue='BAD', kde=True, palette='pastel', bins=30, ax=axes[i])
    axes[i].set_title(f"Distribution of {feature} by Loan Default Status")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")
    axes[i].legend(['Non-Default (0)', 'Default (1)'])

plt.tight_layout()
plt.savefig('./data/Borrower_Characteristics.png')
plt.show()

print("All insights and visualizations consolidated and saved!")