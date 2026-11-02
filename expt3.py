import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load dataset & Exploratory Analysis
df = pd.read_csv('Mall_Customers.csv')
print(df.info())
print(df.describe())

# Select Annual Income (col 3) and Spending Score (col 4)
X = df.iloc[:, [3, 4]].values

# 2. Elbow Method to find optimal K
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# 3. Apply K-Means (k=5) & Visualize Clusters with Centroids
kmeans = KMeans(n_clusters=5, n_init=10, random_state=432)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', label='Customers')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Mall Customer Segments')
plt.legend()
plt.show()

# 4. Interpretation for Targeted Marketing
print("""
Cluster Interpretation:
- Low Income, Low Spend: Cautious customers (offer basic discounts)
- Low Income, High Spend: Impulsive buyers (offer small-ticket promos)
- Mid Income, Mid Spend: Average customers (standard loyalty programs)
- High Income, Low Spend: Careful savers (promote high-value quality items)
- High Income, High Spend: VIP targets (promote luxury/exclusive products)
""")
