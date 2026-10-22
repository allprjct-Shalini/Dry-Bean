import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("1. Loading dataset...")
# Make sure 'Dry_Beans_Dataset.xlsx' is in the same folder
file_path = 'Dry_Beans_Dataset.xlsx'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find {file_path}. Please check your file name!")

df = pd.read_excel(file_path, sheet_name='Dry_Beans_Dataset')

print("2. Cleaning outliers...")
numeric_df = df.drop(columns=['Class'])
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
df_clean = df[mask].copy()

X = df_clean.drop(columns=['Class'])
y = df_clean['Class']

print("3. Encoding and scaling data...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("4. Training Random Forest model...")
best_rf = RandomForestClassifier(n_estimators=100, random_state=42)
best_rf.fit(X_train_scaled, y_train)

print("5. Saving model and preprocessing files...")
joblib.dump(best_rf, 'best_bean_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Success! All 3 .pkl files have been created in your folder:")
print(" - best_bean_model.pkl")
print(" - scaler.pkl")
print(" - label_encoder.pkl")