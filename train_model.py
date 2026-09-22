import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load your specific Excel dataset
df = pd.read_excel('loans.xlsx')

# 2. Map features to the exact column names from your image
X = df[['income', 'credit_score', 'loan_amount', 'employment_years']]
y = df['loan_status']

# 3. Split, scale, and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

print(f"Model trained successfully. Accuracy: {model.score(X_test_scaled, y_test):.2f}")

# 4. Export the trained model and scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model.pkl')