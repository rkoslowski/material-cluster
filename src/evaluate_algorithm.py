import matplotlib.pyplot 	as plt
import numpy 				as np
import pandas 				as pd
from sklearn.cluster 		import KMeans
from sklearn.datasets 		import make_blobs
from sklearn.metrics 		import silhouette_score
from sklearn.preprocessing 	import StandardScaler
from sklearn.cluster 		import DBSCAN
from sklearn.datasets 		import make_moons
from sklearn.metrics 		import adjusted_rand_score
from kneed 					import KneeLocator


cEval = plt.figure(num= "Evaluation of number of clusters", figsize=[13,8])
cEval.tight_layout(pad=3.5)

features, true_labels = make_blobs(
	n_samples=200,
	#n_features=10,
   	centers=3,
	cluster_std=2.75,
	random_state=42
	)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(
	init="random",
	n_clusters=3,
	n_init=10,
	max_iter=100,
	random_state=42
	)

kmeans.fit(scaled_features)

#print(kmeans.inertia_)
#print(kmeans.cluster_centers_)
#print(kmeans.n_iter_)

kmeans_kwargs = {
	"init": "random",
	"n_init": 10,
	"max_iter": 100,
	"random_state": 42
	}


########################### ELBOW METHOD #########################

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
	kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
	kmeans.fit(scaled_features)
	sse.append(kmeans.inertia_)

kl = KneeLocator(
	range(1, 11), sse, curve="convex", direction="decreasing"
	)

print(kl.elbow)

elbowPlot 		= cEval.add_subplot(2, 1, 1)

elbowPlot.set_title("Elbow Method")
elbowPlot.set_xlabel('Number of Clusters')
elbowPlot.set_ylabel('SSE')
elbowPlot.set_xticks(range(1, 11))

elbowPlot.plot(range(1, 11), sse, linestyle='--', marker='o')


########################### SILHOUETTE COEFFICIENT #########################

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 11):
	kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
	kmeans.fit(scaled_features)
	score = silhouette_score(scaled_features, kmeans.labels_)
	silhouette_coefficients.append(score)

silhouettePlot	= cEval.add_subplot(2, 1, 2)

silhouettePlot.plot(range(2, 11), silhouette_coefficients, linestyle='--', marker='o')
silhouettePlot.set_xticks(range(2, 11))
silhouettePlot.set_xlabel("Number of Clusters")
silhouettePlot.set_ylabel("Silhouette Coefficient")


########################### DBSCAN #########################

features, true_labels = make_moons(
    n_samples=250, noise=0.05, random_state=42
)
scaled_features = scaler.fit_transform(features)

# Instantiate k-means and dbscan algorithms
kmeans = KMeans(n_clusters=2)
dbscan = DBSCAN(eps=0.3)

# Fit the algorithms to the features
kmeans.fit(scaled_features)
dbscan.fit(scaled_features)

# Compute the silhouette scores for each algorithm
kmeans_silhouette = silhouette_score(
    scaled_features, kmeans.labels_
).round(2)
dbscan_silhouette = silhouette_score(
   scaled_features, dbscan.labels_
).round (2)

print(kmeans_silhouette)
print(dbscan_silhouette)

# Plot the data and cluster silhouette comparison
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(8, 6), sharex=True, sharey=True
)
fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)
fte_colors = {
    0: "#008fd5",
    1: "#fc4f30",
}
# The k-means plot
km_colors = [fte_colors[label] for label in kmeans.labels_]
ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)
ax1.set_title(
    f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12}
)

# The dbscan plot
db_colors = [fte_colors[label] for label in dbscan.labels_]
ax2.scatter(scaled_features[:, 0], scaled_features[:, 1], c=db_colors)
ax2.set_title(
    f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12}
)

ari_kmeans = adjusted_rand_score(true_labels, kmeans.labels_)
ari_dbscan = adjusted_rand_score(true_labels, dbscan.labels_)

print(round(ari_kmeans, 2))

print(round(ari_dbscan, 2))


plt.show()


