import warnings
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt,\
    matplotlib.mlab as mlab 
from scipy.stats import multivariate_normal
import sklearn.preprocessing as prep
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import StandardScaler
# StandardScaler is a function to normalize the data 
# You may also check MinMaxScaler and MaxAbsScaler 

from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import load_iris

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import silhouette_score

############################################################################
############################################################################

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

def plot_contours(data, means, covs, title):
    k = len(means)
    axs[(k-2)//3 ,(k-2)%3].set_title(f"number of clusters: {k}")
    axs[(k-2)//3 ,(k-2)%3].plot([x[0] for x in data], [y[1] for y in data],\
                                'ko',alpha = 1, markersize=10) # data
    
    delta = 0.025
    x = np.arange(np.amin(np.transpose(data)[0]), np.amax(np.transpose(data)[0]), delta)
    y = np.arange(np.amin(np.transpose(data)[1]), np.amax(np.transpose(data)[1]), delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo','blue','yellow','orange','brown','olive',\
           'pink','crimson']
    for i in range(k):
        mean = means[i]
        if title=="GMM (covariance_type: full)":
            cov = covs[i]
            sigmax = np.sqrt(cov[0][0])
            sigmay = np.sqrt(cov[1][1])
            sigmaxy = cov[0][1]/(sigmax*sigmay)
        elif title=="GMM (covariance_type: tied)":
            cov = covs
            sigmax = np.sqrt(cov[0][0])
            sigmay = np.sqrt(cov[1][1])
            sigmaxy = cov[0][1]/(sigmax*sigmay)
        elif title=="GMM (covariance_type: diag)":
            cov = covs[i]
            sigmax = np.sqrt(cov[0])
            sigmay = np.sqrt(cov[1])
            sigmaxy = 0/(sigmax*sigmay)
        elif title=="GMM (covariance_type: spherical)":
            cov = covs[i]
            sigmax = np.sqrt(cov)
            sigmay = np.sqrt(cov)
            sigmaxy = 0/(sigmax*sigmay)
        rv = multivariate_normal([mean[0], mean[1]],\
                                 [[sigmax, sigmay],\
                                 [sigmaxy, sigmay]])
        Z = rv.pdf(np.dstack((X, Y)))
        axs[(k-2)//3 ,(k-2)%3].contour(X, Y, Z, colors = col[i])
        fig.suptitle(title, fontsize=50, y=1)
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()

############################################################################
############################################################################

dist_type = 'euclidean'
# 'euclidean', 'cosine', 'correlation'

############################################################################
############################################################################

#"""
### Customer Data
customer_data = pd.read_csv("Customer data.csv")
customer_data.drop_duplicates(inplace = True)
customer_data.dropna(inplace = True)
customer_data.set_index(['ID'],inplace = True)
#customer_data.info()
data = customer_data.to_numpy()
scale= StandardScaler()
X = scale.fit_transform(customer_data)
Z = PCA().fit_transform(X)  # with PCA
#Z = X          # without PCA


## Display
fig, axs = plt.subplots(3,2,figsize=(25, 25))
fig.suptitle("Customer Data", fontsize=50, y=0.93)
##
axs[0,0].scatter(np.transpose(data)[4],\
        np.transpose(data)[0],\
            alpha = 1,s=150)
axs[0,0].set_xlabel("Income", fontsize=40)
axs[0,0].set_ylabel("Sex", fontsize=40)
##
axs[0,1].scatter(np.transpose(data)[2],\
        np.transpose(data)[1],\
            alpha = 1,s=150)
axs[0,1].set_xlabel("Age", fontsize=40)
axs[0,1].set_ylabel("Marital Status", fontsize=40)
##
axs[1,0].scatter(np.transpose(data)[4],\
        np.transpose(data)[2],\
            alpha = 1,s=150)
axs[1,0].set_xlabel("Income", fontsize=40)
axs[1,0].set_ylabel("Age", fontsize=40)
##
axs[1,1].scatter(np.transpose(data)[4],\
        np.transpose(data)[5],\
            alpha = 1,s=150)
axs[1,1].set_xlabel("Income", fontsize=40)
axs[1,1].set_ylabel("Occupation", fontsize=40)
##
axs[2,0].scatter(np.transpose(data)[4],\
        np.transpose(data)[3],\
            alpha = 1,s=150)
axs[2,0].set_xlabel("Income", fontsize=40)
axs[2,0].set_ylabel("Education ", fontsize=40)
##
axs[2,1].scatter(np.transpose(data)[4],\
        np.transpose(data)[6],\
            alpha = 1,s=150)
axs[2,1].set_xlabel("Income", fontsize=40)
axs[2,1].set_ylabel("Settlement Size", fontsize=40)


############################################################################

## K-means
metric = np.zeros(9,'d')
silhouette_kmeans = np.zeros(9,'d')-2
K = np.array([2,3,4,5,6,7,8,9,10])
for i in range(9):
    kmeans = KMeans(n_clusters=i+2).fit(Z)
    metric[i] = kmeans.inertia_/len(Z)
    silhouette_kmeans[i] = silhouette_score(Z,\
                                     kmeans.labels_, metric=dist_type)
    if i+2==2 or i+2==3 or i+2==4:
        fig, axs = plt.subplots(3,2,figsize=(25, 25))
        fig.suptitle(f"K-means ({i+2} Clusters)", fontsize=50, y=0.93)
        ##
        axs[0,0].scatter(np.transpose(data)[4],\
                np.transpose(data)[0],\
                    c=kmeans.labels_, alpha = 1,s=150)
        axs[0,0].set_xlabel("Income", fontsize=40)
        axs[0,0].set_ylabel("Sex", fontsize=40)
        ##
        axs[0,1].scatter(np.transpose(data)[2],\
                np.transpose(data)[1],\
                    c=kmeans.labels_,alpha = 1,s=150)
        axs[0,1].set_xlabel("Age", fontsize=40)
        axs[0,1].set_ylabel("Marital Status", fontsize=40)
        ##
        axs[1,0].scatter(np.transpose(data)[4],\
                np.transpose(data)[2],\
                    c=kmeans.labels_,alpha = 1,s=150)
        axs[1,0].set_xlabel("Income", fontsize=40)
        axs[1,0].set_ylabel("Age", fontsize=40)
        ##
        axs[1,1].scatter(np.transpose(data)[4],\
                np.transpose(data)[5],\
                    c=kmeans.labels_,alpha = 1,s=150)
        axs[1,1].set_xlabel("Income", fontsize=40)
        axs[1,1].set_ylabel("Occupation", fontsize=40)
        ##
        axs[2,0].scatter(np.transpose(data)[4],\
                np.transpose(data)[3],\
                    c=kmeans.labels_,alpha = 1,s=150)
        axs[2,0].set_xlabel("Income", fontsize=40)
        axs[2,0].set_ylabel("Education ", fontsize=40)
        ##
        axs[2,1].scatter(np.transpose(data)[4],\
                np.transpose(data)[6],\
                    c=kmeans.labels_,alpha = 1,s=150)
        axs[2,1].set_xlabel("Income", fontsize=40)
        axs[2,1].set_ylabel("Settlement Size", fontsize=40)
plt.figure()
plt.plot(K,metric)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Metric")
plt.title("K-means of Custome Data")
plt.figure()
plt.plot(K,silhouette_kmeans)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Silhouette Score")
plt.title("K-means of Customer Data")


############################################################################

## Agglomerative Clustering
K = np.array([2,3,4,5,6,7,8,9,10])

# linkage: average + distance_threshold
threshold = np.array([3.8,4,4.2,5])  # case with PCA
#threshold = np.array([1,1.8,2.9,3.5])  # case with normalization
fig, axs = plt.subplots(2,2,figsize=(25, 25))
fig.suptitle("Agglomerative Clustering \n (linkage: average, distance_threshold, Income v. Sex)"\
             ,fontsize=50, y=0.96)
silhouette_agglomerative_threshold = np.zeros(4,'d')-2
for i in range(4):
    agglomerative= AgglomerativeClustering(n_clusters = None,\
                                           distance_threshold=threshold[i],\
                                           linkage='average',\
                                           affinity=dist_type)\
                                            .fit(Z)
    silhouette_agglomerative_threshold[i] = silhouette_score(Z,\
                                     agglomerative.labels_, metric=dist_type)
    axs[i//2,i%2].scatter(np.transpose(data)[4],\
            np.transpose(data)[0],\
                c=agglomerative.labels_,s=150)
    axs[i//2 ,i%2].set_title(f"distance_threshold: {threshold[i]}")

###################################

# linkage: average
fig, axs = plt.subplots(3,3,figsize=(25, 25))
fig.suptitle("Agglomerative Clustering \n (linkage: average, Income v. Sex)", \
             fontsize=50, y=0.96)
silhouette_agglomerative_average = np.zeros(9,'d')-2
for i in range(9):
    agglomerative= AgglomerativeClustering(n_clusters=i+2,linkage='average',\
                                           affinity=dist_type)\
                                            .fit(Z)
    silhouette_agglomerative_average[i] = silhouette_score(Z,\
                                     agglomerative.labels_, metric=dist_type)
    axs[i//3,i%3].scatter(np.transpose(data)[4],\
            np.transpose(data)[0],\
                c=agglomerative.labels_,s=150)
    axs[i//3 ,i%3].set_title(f"number of clusters: {i+2}")
dendrogram=hierarchy.linkage(Z,method='average',metric=dist_type)
plt.figure()
dn = hierarchy.dendrogram(dendrogram)
plt.title("Dendrogram of Customer Data (linkage: average)")

###################################

# linkage: single
fig, axs = plt.subplots(3,3,figsize=(25, 25))
fig.suptitle("Agglomerative Clustering \n (linkage: single, Income v. Sex)", \
             fontsize=50, y=0.96)
silhouette_agglomerative_single = np.zeros(9,'d')-2
for i in range(9):
    agglomerative= AgglomerativeClustering(n_clusters=i+2,linkage='single',\
                                           affinity=dist_type)\
                                            .fit(Z)
    silhouette_agglomerative_single[i] = silhouette_score(Z,\
                                     agglomerative.labels_, metric=dist_type)
    axs[i//3,i%3].scatter(np.transpose(data)[4],\
            np.transpose(data)[0],\
                c=agglomerative.labels_,s=150)
    axs[i//3 ,i%3].set_title(f"number of clusters: {i+2}")
dendrogram =hierarchy.linkage(Z,method='single',metric=dist_type)
plt.figure()
dn = hierarchy.dendrogram(dendrogram)
plt.title("Dendrogram of Customer Data (linkage: single)")

###################################

# linkage: ward (only with euclidean distance)
fig, axs = plt.subplots(3,3,figsize=(25, 25))
fig.suptitle("Agglomerative Clustering \n (linkage: ward, Income v. Sex)", \
             fontsize=50, y=0.96)
silhouette_agglomerative_ward = np.zeros(9,'d')-2
for i in range(9):
    agglomerative= AgglomerativeClustering(n_clusters=i+2,linkage='ward',\
                                           affinity='euclidean')\
                                            .fit(Z)
    silhouette_agglomerative_ward[i] = silhouette_score(Z,\
                                     agglomerative.labels_, metric='euclidean')
    axs[i//3,i%3].scatter(np.transpose(data)[4],\
            np.transpose(data)[0],\
                c=agglomerative.labels_,s=150)
    axs[i//3 ,i%3].set_title(f"number of clusters: {i+2}")
dendrogram =hierarchy.linkage(Z,method='ward',metric='euclidean')
plt.figure()
dn = hierarchy.dendrogram(dendrogram)
plt.title("Dendrogram of Customer Data (linkage: ward)")

###################################

plt.figure()
plt.plot(threshold,silhouette_agglomerative_threshold)
plt.xlabel("Distance Threshold")
plt.ylabel("Silhouette Score")
plt.title("Agglomerative Clustering of Customer Data (linkage: average)")

plt.figure()
plt.plot(K,silhouette_agglomerative_average)
plt.plot(K,silhouette_agglomerative_single)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Silhouette Score")
plt.title("Agglomerative Clustering of Customer Data")
plt.legend(['average','single'])


# when using eucliedan dist
plt.figure()
plt.plot(K,silhouette_agglomerative_average)
plt.plot(K,silhouette_agglomerative_single)
plt.plot(K,silhouette_agglomerative_ward)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Silhouette Score")
plt.title("Agglomerative Clustering of Customer Data")
plt.legend(['average','single','ward'])

############################################################################

## DBSCAN
fig, axs = plt.subplots(3,3,figsize=(25, 25))
fig.suptitle("DBSCAN Clustering (Age v. Marital Status)", fontsize=50, y=0.93)
EPS = np.linspace(0.05, 0.1, num=3)
Min_samples = np.linspace(1, 30, num=3)
silhouette_dbscan = np.zeros(15,'d')-2
count = 0
for i in range(3):
    for j in range(3):
        dbscan = DBSCAN(eps=EPS[i], min_samples=Min_samples[j], \
                        metric=dist_type).fit(Z)
        axs[j ,i%3].scatter(np.transpose(data)[2],\
                    np.transpose(data)[1],\
                        c=dbscan.labels_,s=250)
        axs[j ,i%3].title.set_text(\
                            f"eps: {EPS[i]}; min_samples: {Min_samples[j]}")
        if np.sum(dbscan.labels_)>0:
            silhouette_dbscan[count] = silhouette_score(Z,\
                                        dbscan.labels_, metric=dist_type)
        print(f"EPS:{EPS[i]}")
        print(f"Min_samples: {Min_samples[j]}")
        print(f"Silhouette: {silhouette_dbscan[count]}")
        print()
        count += 1
best_score3 = np.amax(silhouette_dbscan)
best_score3_index = np.argmax(silhouette_dbscan)
##
fig, axs = plt.subplots(3,3,figsize=(25, 25))
fig.suptitle("DBSCAN Clustering (Income v. Age)", fontsize=50, y=0.93)
for i in range(3):
    for j in range(3):
        dbscan = DBSCAN(eps=EPS[i], min_samples=Min_samples[j], \
                        metric=dist_type).fit(Z)
        axs[j ,i%3].scatter(np.transpose(data)[4],\
                    np.transpose(data)[2],\
                        c=dbscan.labels_,s=250)
        axs[j ,i%3].title.set_text(\
                            f"eps: {EPS[i]}; min_samples: {Min_samples[j]}")
##
fig, axs = plt.subplots(3,3,figsize=(25, 25))
fig.suptitle("DBSCAN Clustering (Income v. Education)", fontsize=50, y=0.93)
for i in range(3):
    for j in range(3):
        dbscan = DBSCAN(eps=EPS[i], min_samples=Min_samples[j], \
                        metric=dist_type).fit(Z)
        axs[j ,i%3].scatter(np.transpose(data)[4],\
                    np.transpose(data)[3],\
                        c=dbscan.labels_,s=250)
        axs[j ,i%3].title.set_text(\
                            f"eps: {EPS[i]}; min_samples: {Min_samples[j]}")

############################################################################

## Gaussian Mixture (distance_type only changes silhouette)
K = np.array([2,3,4,5,6,7,8,9,10])

#fig, axs = plt.subplots(3,3,figsize=(25, 25))
silhouette_gm_full = np.zeros(9,'d')-2
for i in range(9):
    gm_full = GaussianMixture(n_components=i+2).\
        fit(Z)
    #plot_contours(Z, gm_full.means_, gm_full.covariances_,\
    #              "GMM (covariance_type: full)")
    gm_full_labels = gm_full.fit_predict(Z)
    silhouette_gm_full[i] = silhouette_score(Z,\
                                     gm_full_labels, metric=dist_type)

###################################

#fig, axs = plt.subplots(3,3,figsize=(25, 25))
silhouette_gm_tied = np.zeros(9,'d')-2
for i in range(9):
    gm_tied = GaussianMixture(n_components=i+2,covariance_type='tied').\
        fit(Z)
    #plot_contours(Z, gm_tied.means_, gm_tied.covariances_,\
    #              "GMM (covariance_type: tied)")
    gm_tied_labels = gm_tied.fit_predict(Z)
    silhouette_gm_tied[i] = silhouette_score(Z,\
                                     gm_tied_labels, metric=dist_type)

###################################

#fig, axs = plt.subplots(3,3,figsize=(25, 25))
silhouette_gm_diag = np.zeros(9,'d')-2
for i in range(9):
    gm_diag = GaussianMixture(n_components=i+2,covariance_type='diag').\
        fit(Z)
    #plot_contours(Z, gm_diag.means_, gm_diag.covariances_,\
    #              "GMM (covariance_type: diag)")
    gm_diag_labels = gm_diag.fit_predict(Z)
    silhouette_gm_diag[i] = silhouette_score(Z,\
                                     gm_diag_labels, metric=dist_type)

###################################
        
#fig, axs = plt.subplots(3,3,figsize=(25, 25))
silhouette_gm_spherical = np.zeros(9,'d')-2
for i in range(9):
    gm_spherical = GaussianMixture(n_components=i+2,covariance_type\
                                   ='spherical').fit(Z)
    #plot_contours(Z, gm_spherical.means_, gm_spherical.covariances_,\
    #              "GMM (covariance_type: spherical)")
    gm_spherical_labels = gm_spherical.fit_predict(Z)
    silhouette_gm_spherical[i] = silhouette_score(Z,\
                                     gm_spherical_labels, metric=dist_type)

###################################

plt.figure()
plt.plot(K,silhouette_gm_full)
plt.plot(K,silhouette_gm_tied)
plt.plot(K,silhouette_gm_diag)
plt.plot(K,silhouette_gm_spherical)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Silhouette Score")
plt.title("GMM Clustering of Customer Data")
plt.legend(['full','tied','diag','spherical'])
#"""
