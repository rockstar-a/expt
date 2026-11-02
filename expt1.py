# 1. temperature
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
df = pd.read_csv('temperatures.csv')

# Features (Year) and Targets (All 12 Months)
X = df[['YEAR']]
y = df.loc[:, 'JAN':'DEC']

# I. Train Linear Regression & Predict for inputted year
model = LinearRegression()
model.fit(X, y)

year = int(input("Enter Year to predict: "))
pred_year = model.predict([[year]])
print(f"\nPredicted Month-wise Temperatures for {year}:")
print(pd.Series(pred_year[0], index=y.columns))

# II. Assess performance using MSE and MAE
y_pred = model.predict(X)
print("\nMSE:", mean_squared_error(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))

# III. Visualize Simple Regression Model (for JAN)
plt.scatter(X, df['JAN'], label='Actual JAN Temp')
plt.plot(X, y_pred[:, 0], color='red', label='Regression Line')
plt.xlabel('Year')
plt.ylabel('Temperature (Celsius)')
plt.title('Linear Regression: Year vs JAN Temperature')
plt.legend()
plt.show()
