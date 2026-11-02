# 2. Titanic
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 1. Load data & identify missing values
df = pd.read_csv('train.csv')
print("Missing Values:\n", df.isnull().sum())

# 2. Analyze distributions (Age, Sex, Pclass, Fare, Survived)
for col in ['Age', 'Sex', 'Pclass', 'Fare', 'Survived']:
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# 3. Relationship between Survival and Passenger Characteristics
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival by Gender')
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

# 4. Outliers & Correlation Heatmap
sns.boxplot(x=df['Fare'])
plt.title('Outliers in Fare')
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 5. Predict Probability of Survival (Sex, Age, Pclass)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df[['Sex', 'Age', 'Pclass']]
y = df['Survived']

model = LogisticRegression()
model.fit(X, y)

# Example: Female (1), Age 25, Class 1
prob = model.predict_proba([[1, 25, 1]])[0][1]
print(f"Predicted Survival Probability: {prob:.2f}")
