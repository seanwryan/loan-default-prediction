# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
data_path = './data/hmeq_cleaned.csv'  # Adjust path if needed
df = pd.read_csv(data_path)

# Target variable distribution
print("Target Variable Distribution:")
print(df['BAD'].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
sns.countplot(x='BAD', data=df)
plt.title("Distribution of Loan Defaults (BAD)")
plt.xlabel("Default Status")
plt.ylabel("Count")
plt.xticks([0, 1], ['Non-Default (0)', 'Default (1)'])
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Loan amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['LOAN'], bins=30, kde=True)
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount (Scaled)")
plt.ylabel("Frequency")
plt.show()

# Loan amount vs Default Status
plt.figure(figsize=(8, 5))
sns.boxplot(x='BAD', y='LOAN', data=df)
plt.title("Loan Amount vs Default Status")
plt.xlabel("Default Status")
plt.ylabel("Loan Amount (Scaled)")
plt.xticks([0, 1], ['Non-Default (0)', 'Default (1)'])
plt.show()

# Debt-to-Income Ratio Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['DEBTINC'], bins=30, kde=True)
plt.title("Debt-to-Income Ratio Distribution")
plt.xlabel("Debt-to-Income Ratio (Scaled)")
plt.ylabel("Frequency")
plt.show()

# Debt-to-Income Ratio vs Default Status
plt.figure(figsize=(8, 5))
sns.boxplot(x='BAD', y='DEBTINC', data=df)
plt.title("Debt-to-Income Ratio vs Default Status")
plt.xlabel("Default Status")
plt.ylabel("Debt-to-Income Ratio (Scaled)")
plt.xticks([0, 1], ['Non-Default (0)', 'Default (1)'])
plt.show()