# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("advertising.csv")

print(df.head())

# -----------------------------------
# Data Cleaning
df = df.dropna()

# -----------------------------------
# Feature Selection
X = df.drop("Sales", axis=1)
y = df["Sales"]

# -----------------------------------
# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------
# Model Training (Multiple Linear Regression)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------------
# Prediction
y_pred = model.predict(X_test)

# -----------------------------------
# Evaluation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score:", round(r2, 4))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

# -----------------------------------
# Actual vs Predicted
plt.figure()
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# -----------------------------------
# Feature Importance (coefficients)
coeff = pd.DataFrame({
    "Feature": X.columns,
    "Impact": model.coef_
}).sort_values(by="Impact", ascending=False)

plt.figure()
sns.barplot(x="Impact", y="Feature", data=coeff)
plt.title("Impact of Advertising Channels on Sales")
plt.show()

print("\nFeature Impact:\n", coeff)

# -----------------------------------
# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()