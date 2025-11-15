import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib, os

df = pd.read_csv('data/loan_data.csv')

target = 'loan_status'
y = df[target]
X = df.drop(columns=[target])

for col in X.columns:
    if X[col].dtype == 'object':
        X[col].fillna(X[col].mode()[0], inplace=True)
    elif X[col].dtype == 'bool':
        X[col].fillna(X[col].mode()[0], inplace=True)
    else:
        X[col].fillna(X[col].mean(), inplace=True)

boolean_cols = X.select_dtypes(include=['bool']).columns.tolist()
for col in boolean_cols:
    X[col] = X[col].astype(int)

label_encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

scaler = StandardScaler()
all_cols = X.columns.tolist()  
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=all_cols)

os.makedirs('models', exist_ok=True)
joblib.dump(label_encoders, 'models/encoders.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(categorical_cols, 'models/categorical_columns.joblib')
joblib.dump(boolean_cols, 'models/boolean_columns.joblib')
joblib.dump(X.columns.tolist(), 'models/feature_columns.joblib')

X.to_csv('data/X.csv', index=False)
y.to_csv('data/y.csv', index=False)

print("✅ Preprocessing complete.")
print(f"Total features: {len(X.columns)}")
print(f"Categorical columns (string): {categorical_cols}")
print(f"Boolean columns (True/False): {boolean_cols}")
print(f"Feature columns: {X.columns.tolist()}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

