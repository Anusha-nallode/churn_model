import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("C:/Users/Dell/Downloads/Customer_Churn.csv")

# Drop irrelevant columns
df = df.drop(columns=["customerID"])  # Remove customerID

# Handle missing or blank values in 'TotalCharges'
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # Convert to numeric
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)  # Fill NaNs with median

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'PhoneService',
                                 'MultipleLines', 'InternetService', 'OnlineSecurity',
                                 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                 'StreamingTV', 'StreamingMovies', 'Contract',
                                 'PaperlessBilling', 'PaymentMethod'], drop_first=True)

# Print column names after encoding
print("Features used for training:", df.columns.tolist())

# Define features (X) and target (y)
X = df.drop(columns=["Churn"])  # Remove target column
y = df["Churn"]  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "customer_churn_model.pkl")
print("Model saved successfully!")
