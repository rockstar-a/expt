# 1. Temperatures in India - Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("temperatures.csv")
df.columns = df.columns.str.strip().str.upper()

months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
Y = df[months].apply(pd.to_numeric, errors='coerce').mean().values
X = np.arange(1, 13).reshape(-1, 1)

# Train model and predict
model = LinearRegression().fit(X, Y)
predicted = model.predict(X)

# Evaluate model
print("MSE:", mean_squared_error(Y, predicted))
print("MAE:", mean_absolute_error(Y, predicted))
print("R2 Score:", r2_score(Y, predicted))
print("Equation: Y =", model.intercept_, "+", model.coef_[0], "* X")

# Visualize results
plt.scatter(X, Y, color='blue', label='Actual')
plt.plot(X, predicted, color='red', label='Predicted')
plt.xticks(X.flatten(), months)
plt.xlabel("Months")
plt.ylabel("Temperature (°C)")
plt.title("India's Monthly Temperature - Linear Regression")
plt.legend()
plt.grid()
plt.show()

# Compare and save results
result = pd.DataFrame({
    'Month': months,
    'Actual': Y,
    'Predicted': predicted
})
print(result.round(2))
result.to_csv("temperature_predictions.csv", index=False)
