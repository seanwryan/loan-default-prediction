# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load the cleaned dataset
data_path = './data/hmeq_cleaned.csv'  # Adjust path if necessary
df = pd.read_csv(data_path)

print("Initial Shape of Dataset:", df.shape)

# Step 1: Feature Transformation - Handling Outliers
print("Handling outliers...")
# Clipping outliers in LOAN and DEBTINC (example thresholds)
df['LOAN'] = df['LOAN'].clip(lower=df['LOAN'].quantile(0.01), upper=df['LOAN'].quantile(0.99))
df['DEBTINC'] = df['DEBTINC'].clip(lower=df['DEBTINC'].quantile(0.01), upper=df['DEBTINC'].quantile(0.99))

# Step 2: Interaction Features
print("Creating interaction features...")
# Debt-to-Value Ratio (DEBTINC scaled by VALUE)
df['Debt_to_Value'] = df['DEBTINC'] / (df['VALUE'] + 1e-5)  # Avoid division by zero

# Step 3: Aggregation Features
print("Creating aggregated features...")
# Total derogatory + delinquent records
df['Total_Derog_Delinquent'] = df['DEROG'] + df['DELINQ']

# Step 4: Log Transformation
print("Applying log transformations...")
# Apply log transformation to positively skewed variables
log_cols = ['LOAN', 'MORTDUE', 'VALUE', 'DEBTINC']
for col in log_cols:
    df[col] = np.log1p(df[col])  # log1p handles log(0) gracefully

# Step 5: Feature Binning
print("Creating bins for CLAGE...")
# Create age categories for CLAGE (Credit Line Age)
df['CLAGE_Binned'] = pd.cut(df['CLAGE'], bins=[0, 100, 200, 300, np.inf], labels=['<100', '100-200', '200-300', '>300'])

# Step 6: Handle Missing Values
print("Imputing missing values...")

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Impute missing values for numerical columns
num_imputer = SimpleImputer(strategy='median')
df[numerical_cols] = pd.DataFrame(num_imputer.fit_transform(df[numerical_cols]), columns=numerical_cols)

# Handle categorical columns via one-hot encoding
print("Encoding categorical variables...")
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Step 7: Save the Engineered Dataset
engineered_data_path = './data/hmeq_engineered.csv'
df.to_csv(engineered_data_path, index=False)

print(f"Feature engineering complete. Engineered data saved to {engineered_data_path}.")