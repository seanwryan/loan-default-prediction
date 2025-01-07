# data_analysis.py
import pandas as pd

# Load the dataset
data_path = './data/hmeq.csv'  # Adjust path if needed
df = pd.read_csv(data_path)

# Display basic information about the dataset
print("Dataset Overview:\n")
print(df.head(), "\n")

print("Dataset Summary:\n")
print(df.info(), "\n")

# Check for missing values
print("Missing Values:\n")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0], "\n")

# Check basic statistics
print("Dataset Statistics:\n")
print(df.describe(), "\n")

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}\n")

# Check for class imbalance in the target variable
if 'BAD' in df.columns:
    print("Target Variable Distribution:\n")
    print(df['BAD'].value_counts(normalize=True) * 100)

# Save a summary of missing values to a CSV for easy reference
missing_values.to_csv('./data/missing_values_summary.csv', header=True)
print("Summary of missing values saved to './data/missing_values_summary.csv'.")