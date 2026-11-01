# 2. Titanic Survival Prediction - Logistic Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("train.csv")

# Basic EDA
print(df.head())
print(df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Distributions
for col in ['Age', 'Fare', 'Sex', 'Pclass', 'Survived']:
    sns.histplot(df[col], kde=True) if col in ['Age', 'Fare'] else sns.countplot(x=col, data=df)
    plt.title(col)
    plt.show()

# Survival analysis
sns.countplot(x='Sex', hue='Survived', data=df)
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.show()

sns.boxplot(x='Survived', y='Age', data=df)
plt.show()

# Outliers
sns.boxplot(data=df[['Age', 'Fare']])
plt.show()

# Correlation heatmap
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.show()

# Survival prediction using Sex, Age and Pclass
X = df[['Sex', 'Age', 'Pclass']]
y = df['Survived']

preprocess = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['Sex']),
    (SimpleImputer(strategy='median'), ['Age', 'Pclass'])
)

model = make_pipeline(preprocess, LogisticRegression())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Predict survival probability
passenger = pd.DataFrame({
    'Sex': ['female'],
    'Age': [25],
    'Pclass': [1]
})

probability = model.predict_proba(passenger)[0][1]
print("Survival Probability:", round(probability * 100, 2), "%")
