from typing import Generator
from matplotlib import scale
import matplotlib.pyplot 	as plt
from pandas.core.dtypes.missing import isneginf_scalar
import numpy 				as np
import pandas 				as pd
import seaborn 				as sns
import colorsys
from sklearn.cluster 		import KMeans, DBSCAN
from sklearn.datasets 		import make_blobs, make_moons
from sklearn.decomposition 	import PCA
from sklearn.metrics 		import adjusted_rand_score, silhouette_score
from sklearn.preprocessing 	import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.pipeline       import Pipeline
from kneed 					import KneeLocator

########################### COLOR RANGE ###########################

# return a evenly spaced colorrange
def get_colors(num_colors):
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))

    return colors

########################### DATA PREP ###########################

# Read data and format for sklearn setup
df2 = pd.read_csv('out/twoProps/uniqueValues.csv', sep=';')

#extract Data
dfData =df2.iloc[:,0:3].rename({'Name':'Sample','Density (g/cc)':'Density','E-Modul (GPa)': 'ModulusOfElasticity'}, axis=1)
dfData.to_csv('out/dataAndLabel/data.csv', ';', columns=['Sample','Density', 'ModulusOfElasticity'], index_label=False , index=False)

#extract label
dfLabel=df2[['Name','Class']].rename({'Name':'Sample','Class':'Label'}, axis=1)
dfLabel.to_csv('out/dataAndLabel/labels.csv', ';', columns=['Sample','Label'], index_label=False , index=False)

########################### READ DATA ###########################

datafile 	= "out/dataAndLabel/data.csv"
labels_file = "out/dataAndLabel/labels.csv"

# The KMeans class in scikit-learn requires a NumPy array as an argument.
data = np.genfromtxt(
    datafile,
    delimiter=";",
    usecols=range(1, dfData.shape[1]),
    skip_header=1
)

true_label_names = np.genfromtxt(
    labels_file,
    delimiter=";",
    usecols=(1,),
    skip_header=1,
    dtype="str"
)

########################### LABEL MAPPING ###########################

label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(true_label_names)
#print(label_encoder.classes_)

#print(true_labels)

n_clusters = 6

len(label_encoder.classes_)

########################### CLUSTER BEGIN ###########################

cEval = plt.figure(num= "Evaluation of number of clusters", figsize=[13,8])
cEval.tight_layout(pad=3.5)

scaler = MinMaxScaler()                             # MinMaxScaler / StandardScaler
scaled_features = scaler.fit_transform(data)


kmeans = KMeans(   n_clusters=n_clusters, init="k-means++",       # random / k-means++ 
                    n_init=50,
                    max_iter=250
                    )

kmeans.fit(scaled_features)

#print(kmeans.inertia_)
#print(kmeans.cluster_centers_)
#print(kmeans.n_iter_)

########################### PIPELINES ###########################

pca = PCA(n_components=2)

kmeans_kwargs = {
	"init": "k-means++",       
	"n_init": 50,
	"max_iter": 250,
	}

########################### ELBOW METHOD #########################

maxIters = 31

# A list holds the SSE values for each k
sse = []
for k in range(1, maxIters):
	kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
	kmeans.fit(scaled_features)
	sse.append(kmeans.inertia_)

kl = KneeLocator(
	range(1, maxIters), sse, curve="convex", direction="decreasing"
	)

print("The elbow method suggests", kl.elbow , "clusters!")

elbowPlot 		= cEval.add_subplot(2, 1, 1)

elbowPlot.set_title("Elbow Method")
elbowPlot.set_xlabel('Number of Clusters')
elbowPlot.set_ylabel('SSE')
elbowPlot.set_xticks(range(1, maxIters))

elbowPlot.plot(range(1, maxIters), sse, linestyle='--', marker='o')


########################### SILHOUETTE COEFFICIENT #########################

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, maxIters):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)

maxScore = max(silhouette_coefficients)

print("The silhouette coefficient suggests", silhouette_coefficients.index(maxScore)+2 , "clusters!")

silhouettePlot	= cEval.add_subplot(2, 1, 2)

silhouettePlot.plot(range(2, maxIters), silhouette_coefficients, linestyle='--', marker='o')
silhouettePlot.set_xticks(range(2, maxIters))
silhouettePlot.set_xlabel("Number of Clusters")
silhouettePlot.set_ylabel("Silhouette Coefficient")

n_cs = 6         #kl.elbow
# set number of clusters ( kl.elbow )
kmeans = KMeans(n_clusters=n_cs, **kmeans_kwargs)
kmeans.fit(scaled_features)

dbscan = DBSCAN(eps=0.3)
dbscan.fit(scaled_features)


# The k-means plot

# Plot the data and cluster silhouette comparison
fig, (ax1) = plt.subplots(
    1, 1, figsize=(13, 8), sharex=True, sharey=True
)
fig.suptitle("Clustering Algorithm: K-Means", fontsize=16)

colors = get_colors(n_cs+1)

km_colors = [colors[label] for label in kmeans.labels_]
ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors, alpha=0.7, s=30, edgecolors='face', label=true_label_names)
# mark cluster center as black x
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', s=80)
ax1.set_xlabel("Density")
ax1.set_ylabel("E-Modulus")

########################### DATA PROCESSING ###########################

predicted_labels = kmeans.labels_

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


pcadf = pd.DataFrame(
    data=scaled_features,   
    columns=["den", "emod"],
)


#pcadf["den"] = NormalizeData(pcadf["den"])
#pcadf["emod"] = NormalizeData(pcadf["emod"])

pcadf["predicted_cluster"] = kmeans.labels_
pcadf["true_label"] = label_encoder.inverse_transform(true_labels)


#plt.style.use("bmh")
#plt.figure(figsize=(13, 8),dpi=300.0)
figgg = plt.figure(figsize=(13, 8))

colorDict = {}
for i in range(len(np.unique(kmeans.labels_))):
    colorDict[i] = colors[i]

scat = sns.scatterplot(
    "den",
    "emod",
    s=50,
    data=pcadf,
    hue="predicted_cluster",
    style="true_label",
    palette=colorDict,
    #size_norm=(0,1)
)

scat = sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, marker='x', color='black')
scat.set_title('Clustering results Material Data')
scat.set_xlabel('Density [g/cc]')
scat.set_ylabel('E-Modul [GPa]')

#plt.legend(bbox_to_anchor=(1.05,1),loc=2, borderaxespad=0.0)
plt.legend(loc='upper right')
plt.show()

figgg.savefig(fname='testout', dpi=300)

exit()
########################### PIPELINES ###########################

preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),     # MinMaxScaler / StandardScaler
        ("pca", PCA(n_components=2)),
    ]
)

clusterer_k_means = Pipeline(
   [
       (
           "kmeans",
           KMeans(
               n_clusters=n_clusters,
               init="k-means++",       # random / k-means++ 
               n_init=50,
               max_iter=250,
           ),
       ),
   ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer_k_means)
    ]
)


########################### DATA PROCESSING ###########################

# with data as the argument, fit() performs all the pipeline steps on the data
pipe.fit(data)

preprocessed_data = pipe["preprocessor"].transform(data)

predicted_labels = pipe["clusterer"]["kmeans"].labels_

print(silhouette_score(preprocessed_data, predicted_labels))

print(adjusted_rand_score(true_labels, predicted_labels))

pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns=["den", "emod"],
)

pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

#plt.style.use("bmh")
plt.figure(figsize=(13, 8))

scat = sns.scatterplot(
    "den",
    "emod",
    s=50,
    data=pcadf,
    hue="predicted_cluster",
    style="true_label",
    palette="Set2",
)

scat.set_title('Clustering results Material Data')
scat.set_xlabel('Density [g/cc]')
scat.set_ylabel('E-Modul [GPa]')

#scat.set_yscale('log')
#scat.set_xscale('log')

plt.legend(bbox_to_anchor=(1.05,1),loc=2, borderaxespad=0.0)

plt.show()