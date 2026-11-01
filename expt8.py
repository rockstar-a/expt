# 8. Real estate price prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro

# Load dataset (first 50 houses)
df = pd.read_csv("Housing.csv").head(50)

# Select required columns
price = df['price']
area = df['area']
bedrooms = df['bedrooms']

# Statistical analysis
print("Mean Price:", price.mean())
print("Median Price:", price.median())
print("Variance:", price.var())
print("Standard Deviation:", price.std())
print("Average Bedrooms:", bedrooms.mean())

# Correlation
print("\nArea-Price Correlation:", area.corr(price))

# Scatter plot
plt.scatter(area, price)
plt.xlabel("Carpet Area")
plt.ylabel("House Price")
plt.title("Area vs House Price")
plt.show()

# Histogram with normal distribution curve
plt.hist(price, bins=10, density=True, alpha=0.6, edgecolor='black')
x = np.linspace(price.min(), price.max(), 100)
plt.plot(x, norm.pdf(x, price.mean(), price.std()), 'r')
plt.xlabel("House Price")
plt.ylabel("Density")
plt.title("House Price Distribution")
plt.show()

# Normality test
stat, p = shapiro(price)
print("\nShapiro Test p-value:", p)
print("Normally Distributed" if p > 0.05 else "Not Normally Distributed")

# Interpretation
print("\nInterpretation:")
print("Positive correlation: Larger houses tend to have higher prices."
      if area.corr(price) > 0 else "No positive relationship between area and price.")
