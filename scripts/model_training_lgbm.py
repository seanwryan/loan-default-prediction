# model_training_lgbm.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import lightgbm as lgb

# Load the engineered dataset
data_path = './data/hmeq_engineered.csv'  # Adjust path if necessary
df = pd.read_csv(data_path)

# Define features (X) and target (y)
X = df.drop(columns=['BAD'])
y = df['BAD']

# Train-test split
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# LightGBM Parameters
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',  # Define metric explicitly
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'is_unbalance': True,  # Address class imbalance
    'seed': 42
}

# Train LightGBM Model with Callbacks
print("Training LightGBM model...")
callbacks = [
    lgb.early_stopping(stopping_rounds=50),  # Stop if no improvement for 50 rounds
    lgb.log_evaluation(period=100)          # Log metrics every 100 rounds
]

lgbm_model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    valid_names=["Train", "Test"],
    num_boost_round=1000,
    callbacks=callbacks
)

# Predictions
print("Generating predictions...")
y_pred = (lgbm_model.predict(X_test) > 0.5).astype(int)
y_prob = lgbm_model.predict(X_test)

# Evaluation Metrics
print("Evaluation Metrics:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# ROC Curve
print("Generating ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()