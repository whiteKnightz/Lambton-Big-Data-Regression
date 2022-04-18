import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('./Dry_Bean_Dataset.csv')
encoder = LabelEncoder()
df['Class'] = encoder.fit_transform(df['Class'])

points = df.iloc[:, 1:14].values
x = points[:, 0]
y = points[:, 1]
inertias = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(points)
    inertias.append(kmeans.inertia_)
plt.plot(range(1, 10), inertias)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
plt.close()

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(points)
predicted_cluster_indexes = kmeans.predict(points)
plt.scatter(x, y, c=predicted_cluster_indexes, s=50, alpha=0.7, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100)
plt.show()
plt.close()

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(points)
df['Cluster'] = kmeans.predict(points)

results = pd.DataFrame(
    columns=['Cluster', 'Average Area', 'Average Perimeter', 'Average roundness', 'Average Compactness',
             'No. of SEKER', 'No. of BARBUNYA', 'No. of BOMBAY', 'No. of CALI',
             'No. of HOROZ', 'No. of SIRA', 'No. of DERMASON'])
for i in range(len(kmeans.cluster_centers_)):
    area = df[df['Cluster'] == i]['Area'].mean()
    perimeter = df[df['Cluster'] == i]['Perimeter'].mean()
    roundness = df[df['Cluster'] == i]['roundness'].mean()
    compactness = df[df['Cluster'] == i]['Compactness'].mean()
    gdf = df[df['Cluster'] == i]
    SEKER = gdf[gdf['Class'] == 5].shape[0]
    BARBUNYA = gdf[gdf['Class'] == 0].shape[0]
    BOMBAY = gdf[gdf['Class'] == 1].shape[0]
    CALI = gdf[gdf['Class'] == 2].shape[0]
    HOROZ = gdf[gdf['Class'] == 4].shape[0]
    SIRA = gdf[gdf['Class'] == 6].shape[0]
    DERMASON = gdf[gdf['Class'] == 3].shape[0]
    results.loc[i] = (
    [i, area, perimeter, roundness, compactness, SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON])
print(results.head())
