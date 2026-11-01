# 3. Mall Customer Segmentation - K-Means Clustering
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Select and scale features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_scaled = StandardScaler().fit_transform(X)

# Elbow Method
inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Silhouette Scores
for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    print("K =", k, "Score =", round(silhouette_score(X_scaled, labels), 4))

# Apply K-Means with 5 clusters
model = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)

# Cluster visualization
sns.scatterplot(data=df, x=X.columns[0], y=X.columns[1], hue="Cluster", palette="Set1")
plt.scatter(*StandardScaler().fit(X).inverse_transform(model.cluster_centers_).T,
            color='black', marker='X', s=200, label='Centroids')
plt.title("Mall Customer Segmentation")
plt.legend()
plt.show()

# Cluster summary
print("\nCluster Summary:")
print(df.groupby("Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2))

# Save results
df.to_csv("Mall_Customers_Clustered.csv", index=False)
print("Clustering Completed!")
