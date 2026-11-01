# 7. Academic performance sheet using pandas and matplotlib.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and select 30 students
df = pd.read_csv("StudentPerformanceFactors.csv")
df = df[['Hours_Studied', 'Attendance', 'Exam_Score']].dropna()
df = df.sample(30, random_state=42)
df.columns = ['Study_Hours', 'Attendance', 'Marks']

# Academic performance sheet
df.insert(0, 'Student_ID', range(1, 31))
print("\nACADEMIC PERFORMANCE SHEET\n", df)
df.to_csv("Performance_Sheet.csv", index=False)

# Statistical analysis
m = df['Marks']
print("\nMean:", m.mean())
print("Median:", m.median())
print("Variance:", m.var())
print("Standard Deviation:", m.std())
print("Correlation:", df['Study_Hours'].corr(m))

# Graph 1: Study Hours vs Marks
sns.regplot(data=df, x='Study_Hours', y='Marks')
plt.title("Study Hours vs Marks")
plt.show()

# Graph 2: Normal Distribution Curve
sns.histplot(m, kde=True, stat='density')
plt.title("Normal Distribution of Marks")
plt.show()

# Graph 3: Box Plot
sns.boxplot(x=m)
plt.title("Distribution of Marks")
plt.show()

# Graph 4: Attendance vs Marks
sns.scatterplot(data=df, x='Attendance', y='Marks')
plt.title("Attendance vs Marks")
plt.show()

# Interpretation
r = df['Study_Hours'].corr(m)
print("\nINTERPRETATION:")
print("Positive correlation" if r > 0 else "Negative correlation" if r < 0 else "No correlation")
print("Average Marks:", m.mean())
