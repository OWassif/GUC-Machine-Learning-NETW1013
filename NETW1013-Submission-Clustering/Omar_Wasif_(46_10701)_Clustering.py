import warnings
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import sklearn.preprocessing as prep
#from sklearn.preprocessing import StandardScaler
# StandardScaler is a function to normalize the data 
# You may also check MinMaxScaler and MaxAbsScaler 

from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import load_iris

from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import silhouette_score

############################################################################
############################################################################

def display_cluster(X,km=[],num_clusters=0):
    plt.figure()
    color = 'brgcmyk'  #List colors
    alpha = 0.5  #color obaque
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_==i,0],X[km.labels_==i,1],\
                        alpha = alpha,s=s)
            plt.scatter(km.cluster_centers_[i][0],km.cluster_centers_[i][1],\
                        marker = 'x', s = 100)

############################################################################
############################################################################

#"""
# Multi-Blob Data

plt.rcParams['figure.figsize'] = [8,8]
sns.set_style("whitegrid")
sns.set_context("talk")
n_bins = 6  
centers = [(-3, -3), (0, 0), (5,2.5),(-1, 4), (4, 6), (9,7)]
Multi_blob_Data, y = make_blobs(n_samples=[100,150, 300, 400,300, 200],\
                                n_features=2, \
                                    cluster_std=[1.3,0.6, 1.2, 1.7,0.9,1.7],\
                                        centers=centers, \
                                            shuffle=False, random_state=42)
display_cluster(Multi_blob_Data)

###################################

# K-means
fig, axs = plt.subplots(3,3,figsize=(25, 25))
fig.suptitle("K-means Clustering", fontsize=50, y=0.93)
metric = np.zeros(9,'d')
silhouette = np.zeros(9,'d')-2
K = np.array([2,3,4,5,6,7,8,9,10])
for i in range(9):
    kmeans = KMeans(n_clusters=i+2).fit(Multi_blob_Data)
    metric[i] = kmeans.inertia_/len(Multi_blob_Data)
    silhouette[i] = silhouette_score(Multi_blob_Data,\
                                     kmeans.labels_, metric='euclidean')
    #plt.figure()
    #display_cluster(Multi_blob_Data,kmeans,num_clusters=i+2)
    axs[i//3,i%3].scatter(np.transpose(Multi_blob_Data)[0],\
            np.transpose(Multi_blob_Data)[1],\
                c=kmeans.labels_,alpha = 0.75)
    axs[i//3,i%3].scatter(np.transpose(kmeans.cluster_centers_)[0],\
                          np.transpose(kmeans.cluster_centers_)[1],\
                              c='b',marker='x',s=250)
    axs[i//3 ,i%3].set_title(f"number of clusters: {i+2}")
plt.figure()
plt.plot(K,metric)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Metric")
plt.title("K-means of Multi-Blob Data")
plt.figure()
plt.plot(K,silhouette)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Silhouette Score")
plt.title("K-means of Multi-Blob Data")
best_score1 = np.amax(silhouette)
best_score1_index = np.argmax(silhouette)

###################################

# Agglomerative Clustering
fig, axs = plt.subplots(3,3,figsize=(25, 25))
fig.suptitle("Agglomerative Clustering", fontsize=50, y=0.93)
silhouette = np.zeros(9,'d')-2
for i in range(9):
    agglomerative= AgglomerativeClustering(n_clusters=i+2).fit(Multi_blob_Data)
    silhouette[i] = silhouette_score(Multi_blob_Data,\
                                     agglomerative.labels_, metric='euclidean')
    #plt.figure()
    axs[i//3,i%3].scatter(np.transpose(Multi_blob_Data)[0],\
            np.transpose(Multi_blob_Data)[1],\
                c=agglomerative.labels_)
    axs[i//3 ,i%3].set_title(f"number of clusters: {i+2}")
#dendrogram = hierarchy.linkage(Multi_blob_Data)
#plt.figure()
#dn = hierarchy.dendrogram(dendrogram)
plt.figure()
plt.plot(K,silhouette)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Silhouette Score")
plt.title("Agglomerative Clustering of Multi-Blob Data")
best_score2 = np.amax(silhouette)
best_score2_index = np.argmax(silhouette)

###################################

# DBSCAN
fig, axs = plt.subplots(3,5,figsize=(25, 25))
fig.suptitle("DBSCAN Clustering", fontsize=50, y=0.93)
EPS = np.linspace(0.5, 1, num=5)
Min_samples = np.linspace(5, 25, num=5)
silhouette = np.zeros(15,'d')-2
count = 0
for i in range(5):
    for j in range(3):
        dbscan = DBSCAN(eps=EPS[i], min_samples=Min_samples[j]).\
            fit(Multi_blob_Data)
        #plt.figure()
        axs[j ,i%5].scatter(np.transpose(Multi_blob_Data)[0],\
                    np.transpose(Multi_blob_Data)[1],\
                        c=dbscan.labels_)
        axs[j ,i%5].title.set_text(\
                            f"eps: {EPS[i]}; min_samples: {Min_samples[j]}")
        if np.sum(dbscan.labels_)>0:
            silhouette[count] = silhouette_score(Multi_blob_Data,\
                                        dbscan.labels_, metric='euclidean')
        count += 1
best_score3 = np.amax(silhouette)
best_score3_index = np.argmax(silhouette)

###################################

# Gaussian Mixture
K = np.array([2,3,4,5,6,7,8,9,10])
for i in range(9):
    gm = GaussianMixture(n_components=i+2).fit(Multi_blob_Data)
#"""

############################################################################

"""
# Moons Data

n_samples = 1000
Moons, y = noisy_moons = make_moons(n_samples=n_samples, noise= .1)
display_cluster(Moons)

###################################

# K-means
fig, axs = plt.subplots(3,3,figsize=(25, 25))
fig.suptitle("K-means Clustering", fontsize=50, y=0.93)
metric = np.zeros(9,'d')
silhouette = np.zeros(9,'d')-2
K = np.array([2,3,4,5,6,7,8,9,10])
for i in range(9):
    kmeans = KMeans(n_clusters=i+2).fit(Moons)
    #plt.figure()
    axs[i//3,i%3].scatter(np.transpose(Moons)[0],np.transpose(Moons)[1],\
                          c=kmeans.labels_,alpha = 0.75)
    axs[i//3,i%3].scatter(np.transpose(kmeans.cluster_centers_)[0],\
                          np.transpose(kmeans.cluster_centers_)[1],\
                              c='b',marker='x',s=250)
    metric[i] = kmeans.inertia_/len(Moons)
    silhouette[i] = silhouette_score(Moons,\
                                     kmeans.labels_, metric='euclidean')
    axs[i//3 ,i%3].set_title(f"number of clusters: {i+2}")
plt.figure()
plt.plot(K,metric)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Metric")
plt.title("K-means of Moons Data")
plt.figure()
plt.plot(K,silhouette)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Silhouette Score")
plt.title("K-means of Multi-Blob Data")
best_score4 = np.amax(silhouette)
best_score4_index = np.argmax(silhouette)

###################################

# Agglomerative Clustering
fig, axs = plt.subplots(3,3,figsize=(25, 25))
fig.suptitle("Agglomerative Clustering", fontsize=50, y=0.93)
silhouette = np.zeros(9,'d')-2
for i in range(9):
    agglomerative = AgglomerativeClustering(n_clusters=i+2).fit(Moons)
    #plt.figure()
    axs[i//3,i%3].scatter(np.transpose(Moons)[0],\
            np.transpose(Moons)[1],\
                c=agglomerative.labels_)
    axs[i//3 ,i%3].set_title(f"number of clusters: {i+2}")
    silhouette[i] = silhouette_score(Moons,\
                                     agglomerative.labels_, metric='euclidean')
#dendrogram = hierarchy.linkage(Moons)
#plt.figure()
#dn = hierarchy.dendrogram(dendrogram)
plt.figure()
plt.plot(K,silhouette)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Silhouette Score")
plt.title("Agglomerative Clustering of Multi-Blob Data")
best_score5 = np.amax(silhouette)
best_score5_index = np.argmax(silhouette)

###################################

# DBSCAN
fig, axs = plt.subplots(3,3,figsize=(25, 25))
fig.suptitle("DBSCAN Clustering", fontsize=50, y=0.93)
EPS = np.linspace(0.1, 0.2, num=3)
Min_samples = np.linspace(5, 25, num=5)
silhouette = np.zeros(15,'d')-2
count = 0
for i in range(3):
    for j in range(3):
        dbscan = DBSCAN(eps=EPS[i], min_samples=Min_samples[j]).\
            fit(Moons)
        #plt.figure()
        axs[j ,i%3].scatter(np.transpose(Moons)[0],\
                    np.transpose(Moons)[1],\
                        c=dbscan.labels_)
        axs[j ,i%3].title.set_text(\
                            f"eps: {EPS[i]}; min_samples: {Min_samples[j]}")
        if np.sum(dbscan.labels_)>0:
            silhouette[count] = silhouette_score(Moons,\
                                        dbscan.labels_, metric='euclidean')
        count += 1
best_score6 = np.amax(silhouette)
best_score6_index = np.argmax(silhouette)

#"""

############################################################################

"""
customer_data = pd.read_csv("Customer data.csv")
customer_data.drop_duplicates(inplace = True)
customer_data.dropna(inplace = True)
customer_data.set_index(['ID'],inplace = True)
customer_data.info()

X = customer_data.to_numpy()
metric = np.zeros(9,'d')
K = np.array([2,3,4,5,6,7,8,9,10])
for i in range(9):
    kmeans = KMeans(n_clusters=i+2).fit(X)
    metric[i] = Cluster_Metric
plt.figure()
plt.plot(K,metric)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Metric")
plt.title("K-means of Customer Data")
"""
