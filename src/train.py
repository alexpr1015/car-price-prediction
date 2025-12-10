import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
data_path = os.path.join("data", "CarPrice_Assignment.csv")
df = pd.read_csv(data_path)
print(df.head())

# Preprocessing
df.drop(["car_ID", "CarName"], axis=1, inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features & Target
X = df.drop("price", axis=1)
Y = df["price"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# Model evaluation
print("\nModel Evaluation:")
print("R² Score:", r2_score(Y_test, Y_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_test, Y_pred)))

# Create output folder if not exists
os.makedirs("outputs", exist_ok=True)

# Actual vs Predicted plot
plt.figure(figsize=(7, 5))
sns.scatterplot(x=Y_test, y=Y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.savefig(os.path.join("outputs", "Actual_vs_Predicted.png"))
plt.show()

# Feature Importance
coefficients = pd.Series(model.coef_, index=X.columns)
coefficients.sort_values().plot(kind="barh", figsize=(10, 6))
plt.title("Feature Importance - Linear Regression Coefficients")
plt.savefig(os.path.join("outputs", "feature_importance.png"))
plt.show()
