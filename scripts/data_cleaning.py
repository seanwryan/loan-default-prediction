# data_cleaning.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load the dataset
data_path = './data/hmeq.csv'  # Adjust path if necessary
df = pd.read_csv(data_path)

print("Initial Data Shape:", df.shape)

# Handle missing values
print("Handling missing values...")

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Impute missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Encode categorical variables
print("Encoding categorical variables...")
df = pd.get_dummies(df, columns=['REASON', 'JOB'], drop_first=True)

# Normalize numerical variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

num_features = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
df[num_features] = scaler.fit_transform(df[num_features])

# Save the cleaned dataset
cleaned_data_path = './data/hmeq_cleaned.csv'
df.to_csv(cleaned_data_path, index=False)

print(f"Data cleaning completed! Cleaned data saved to {cleaned_data_path}.")