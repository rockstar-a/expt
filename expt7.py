# 7. Student Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load first 30 students showing required columns
df = pd.read_csv('StudentPerformanceFactors.csv').sample(n=30, random_state=100)
print(df[['Hours_Studied', 'Attendance', 'Exam_Score']].head())

# 1. Mean, Median, Variance, Standard Deviation of Exam_Score
print("Mean    :", df['Exam_Score'].mean())
print("Median  :", df['Exam_Score'].median())
print("Variance:", df['Exam_Score'].var())
print("Std Dev :", df['Exam_Score'].std())

# 2. Correlation between Hours_Studied and Exam_Score
print("Correlation:", df['Hours_Studied'].corr(df['Exam_Score']))

# 3. Scatter Plot
plt.scatter(df['Hours_Studied'], df['Exam_Score'])
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Hours Studied vs Exam Score')
plt.show()

# 4. Histogram with Normal Distribution Curve
sns.histplot(df['Exam_Score'], kde=True)
plt.title('Exam Score Distribution with Normal Curve')
plt.show()

# 5. Interpretation
print("Interpretation: Positive correlation shows more study hours lead to higher exam scores. The bell-shaped curve indicates marks are approximately normally distributed.")
