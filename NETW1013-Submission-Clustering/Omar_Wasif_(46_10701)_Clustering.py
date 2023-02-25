import warnings
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import sklearn.preprocessing as prep
from sklearn.datasets import make_blobs
from scipy.cluster import hierarchy

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# StandardScaler is a function to normalize the data 
# You may also check MinMaxScaler and MaxAbsScaler 
#from sklearn.preprocessing import StandardScaler

############################################################################
############################################################################

# helper function that allows us to display data in 2 dimensions an...
#   ...highlights the clusters
def display_cluster(X,km=[],num_clusters=0):
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

# K-means of Multi-Blob Data
metric = np.zeros(9,'d')
K = np.array([2,3,4,5,6,7,8,9,10])
for i in range(9):
    kmeans = KMeans(n_clusters=i+2).fit(Multi_blob_Data)
    plt.figure()
    display_cluster(Multi_blob_Data,kmeans,num_clusters=i+2)
    plt.title("K-Means")
    metric[i] = kmeans.inertia_/len(Multi_blob_Data)
plt.figure()
plt.plot(K,metric)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Metric")
plt.title("K-means of Multi-Blob Data")

# Agglomerative Clustering of Multi-Blob Data
for i in range(9):
    agglomerative = AgglomerativeClustering(n_clusters=i+2).fit(Multi_blob_Data)
    plt.figure()
    plt.scatter(np.transpose(Multi_blob_Data)[0],\
            np.transpose(Multi_blob_Data)[1],\
                c=agglomerative.labels_)
    plt.title("Agglomerative Clustering")
    

# DBSCAN of Multi-Blob Data
EPS = np.linspace(0.1, 3, num=5)
Min_samples = np.linspace(5, 25, num=5)
for i in range(5):
    for j in range(3):
        dbscan = DBSCAN(eps=EPS[i], min_samples=Min_samples[j]).\
            fit(Multi_blob_Data)
        plt.figure()
        plt.scatter(np.transpose(Multi_blob_Data)[0],\
                    np.transpose(Multi_blob_Data)[1],\
                        c=dbscan.labels_)
        plt.title(f"eps: {EPS[i]}; min_samples: {Min_samples[j]}")

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
