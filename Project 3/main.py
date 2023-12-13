import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

df = pd.read_csv('Mall_Customers.csv')
df.head()

from sklearn.cluster import KMeans

X = df.iloc[:, [3, 4]].values

# KMeans clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to the DataFrame
df['Cluster'] = y_kmeans

# Colors for different clusters
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

# Plotting clusters
plt.figure(figsize=(8, 6))
for i in range(5):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data.iloc[:, 3], cluster_data.iloc[:, 4], s=100, c=colors[i], label=f'Cluster {i+1}')

# Plotting centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

import seaborn as sns
# Select columns for clustering
X = df.iloc[:, [3, 4]].values

# KMeans clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to the DataFrame
df['Cluster'] = y_kmeans

# Plotting using seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=80, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids', marker='X')
plt.title('Clusters of customers')
plt.legend()
plt.show()

# Hierarchial Clustering 
# Select columns for clustering
X = df.iloc[:, [3, 4]].values

# Plotting the dendrogram without x-axis labels
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'), leaf_font_size=10, leaf_rotation=90, color_threshold=300, labels=None)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.axhline(y=300, color='black', linestyle='--', label='Threshold for 5 clusters')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-axis ticks and labels
plt.legend(loc='right')
plt.show()
