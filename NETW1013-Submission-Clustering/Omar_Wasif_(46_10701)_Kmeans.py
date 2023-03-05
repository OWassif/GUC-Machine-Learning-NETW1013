import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import math

############################################################################
############################################################################

def GUC_Distance (Cluster_Centroids, Data_points, Distance_Type ):
    
    len_c = len(Cluster_Centroids)
    len_x = len(Data_points)
    
    Cluster_Distance = np.zeros((len_x,len_c),'d') -1
    
    if Distance_Type == 0:
        Cluster_Distance = cdist(Data_points, Cluster_Centroids,'euclidean')
        # for i in range(len_x):
        #     for j in range(len_c):
        #         diff_arr = np.subtract(Cluster_Centroids[j],Data_points[i])
        #         sq_diff_arr = np.square(diff_arr)
        #         Cluster_Distance[i][j] = np.sqrt(np.sum(sq_diff_arr))
    
    if Distance_Type == 1:
        Cluster_Distance = cdist(Data_points, Cluster_Centroids,'correlation')
    
    return Cluster_Distance 

############################################################################

def msd (Assgn_Cluster, Assgn_Cluster_Distance, Number_of_Clusters, len_x ):
    
    mean_square_distance = np.zeros((Number_of_Clusters),'d') # per cluster
    flag = mean_square_distance > 0

    square_distance = np.square(Assgn_Cluster_Distance)
    for i in range(len_x):
        mean_square_distance[Assgn_Cluster[i]] += square_distance[i]
        flag[Assgn_Cluster[i]] = True
    
    return [mean_square_distance, flag]

############################################################################

def initilaize (Data_points, Number_of_Clusters ):
    
    len_f = len(np.transpose(Data_points))
                                        
    Cluster_Centroids = np.transpose(np.zeros((Number_of_Clusters,len_f),'d'))
        
    for i in range(len_f):
        data_feat_arr = np.transpose(Data_points)[i]
        upper = np.max(data_feat_arr)
        lower = np.min(data_feat_arr)
        cluster_feat_arr = np.random.uniform(lower, upper, Number_of_Clusters)
        Cluster_Centroids[i] = cluster_feat_arr
    Cluster_Centroids = np.transpose(Cluster_Centroids)
    
    return Cluster_Centroids 

############################################################################

def update (Data_points,Assgn_Cluster,previous_centroids,Number_of_Clusters):
    
    Cluster_Centroids = np.zeros(np.shape(previous_centroids))

    for i in range(Number_of_Clusters):
        index_arr = np.array(Assgn_Cluster == i)
        cluster_members = Data_points[index_arr]
        if cluster_members.size ==0:
            Cluster_Centroids[i] = previous_centroids[i]
        else:
            cluster_members_sum = np.sum(cluster_members,axis=0)
            new_centroid = np.divide(cluster_members_sum,len(cluster_members))
            Cluster_Centroids[i] = new_centroid
        
    return Cluster_Centroids

############################################################################

def GUC_Kmean ( Data_points, Number_of_Clusters,  Distance_Type ):
    
    len_x = len(Data_points)
    len_f = len(np.transpose(Data_points))
    
    #Final_Cluster_Centroids = np.zeros((Number_of_Clusters,len_f),'d')
    #Final_Assgn_Cluster = np.zeros((len_x),'d') -1
    #Final_Cluster_Distance = np.zeros((len_x),'d') -1
    Cluster_Metric = math.inf
    
    for runs in range(100):
        #Assgn_Cluster = np.zeros((len_x),'d') -1
        #Assgn_Cluster_Distance = np.zeros((len_x),'d') -1
        #Metric = -1
        
        Cluster_Centroids = initilaize(Data_points, Number_of_Clusters)
        previous_centroids = np.add(Cluster_Centroids,1000)
        #print(previous_centroids)
    
        while np.sum(np.absolute(np.subtract(Cluster_Centroids,\
                                             previous_centroids)))>0:
            previous_centroids = Cluster_Centroids
            Cluster_Distance = GUC_Distance(Cluster_Centroids, Data_points, 0)
            Assgn_Cluster = np.argmin(Cluster_Distance,axis=1)
            Assgn_Cluster_Distance =  np.amin(Cluster_Distance,axis=1)
            Cluster_Centroids = update(Data_points, Assgn_Cluster, \
                                       previous_centroids, Number_of_Clusters)
            
        [mean_square_distance, flag] = msd(Assgn_Cluster,\
                                           Assgn_Cluster_Distance,\
                                               Number_of_Clusters, len_x)
        Metric = np.sum(mean_square_distance)/len_x
        
        if Metric <= Cluster_Metric:
            Final_Cluster_Centroids = Cluster_Centroids
            Final_Assgn_Cluster = Assgn_Cluster
            Final_Cluster_Distance = Assgn_Cluster_Distance
            Cluster_Metric = Metric
    
    return [ Final_Cluster_Distance , Cluster_Metric , Final_Cluster_Centroids\
            , Final_Assgn_Cluster ]

############################################################################

# helper function that allows us to display data in 2 dimensions an...
#   ...highlights the clusters
def display_cluster(X,km=[],num_clusters=0):
    plt.figure()
    color = 'brgcmyk'  #List colors
    alpha = 0.5  #color obaque
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_==i,0],X[km.labels_==i,1],c = color[i],\
                        alpha = alpha,s=s)
            plt.scatter(km.cluster_centers_[i][0],km.cluster_centers_[i][1],\
                        c = color[i], marker = 'x', s = 100)

############################################################################
############################################################################

distance_type = 0


# K-means of Circular Data
#"""
# prepare the figure sise and background 
# this part can be replaced by a number of subplots 
plt.rcParams['figure.figsize'] = [8,8]
sns.set_style("whitegrid")
sns.set_context("talk")
# Produce a data set that represent the x and y o coordinates of a circle 
# this part can be replaced by data that you import froma file 
angle = np.linspace(0,2*np.pi,20, endpoint = False)
X = np.append([np.cos(angle)],[np.sin(angle)],0).transpose()
# Data is displayed 
# to display the data only it is assumed that the number of clusters is zero..
#   ...which is the default of the fuction 
display_cluster(X)

fig, axs = plt.subplots(3,3,figsize=(25, 25))
metric = np.zeros(9,'d')
K = np.array([2,3,4,5,6,7,8,9,10])
for i in range(9):
    [ Final_Cluster_Distance , Cluster_Metric , Final_Cluster_Centroids ,\
     Final_Assgn_Cluster ] = GUC_Kmean(X, i+2, distance_type)
    #plt.figure()
    axs[i//3,i%3].scatter(np.transpose(X)[0],np.transpose(X)[1],\
                          c=Final_Assgn_Cluster,s=250)
    axs[i//3,i%3].scatter(np.transpose(Final_Cluster_Centroids)[0],\
                          np.transpose(Final_Cluster_Centroids)[1],\
                              c=np.arange(i+2),marker='x',s=250)
    metric[i] = Cluster_Metric
plt.figure()
plt.plot(K,metric)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Metric")
plt.title("K-means of Circular Data")
#"""

############################################################################

# K-means of Multi-Blob Data
#"""
n_samples = 1000
n_bins = 4  
centers = [(-3, -3), (0, 0), (3, 3), (6, 6), (9,9)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)
display_cluster(X)

fig, axs = plt.subplots(3,3,figsize=(25, 25))
metric = np.zeros(9,'d')
K = np.array([2,3,4,5,6,7,8,9,10])
for i in range(9):
    [ Final_Cluster_Distance , Cluster_Metric , Final_Cluster_Centroids ,\
     Final_Assgn_Cluster ] = GUC_Kmean(X, i+2, distance_type)
    #plt.figure()
    axs[i//3,i%3].scatter(np.transpose(X)[0],np.transpose(X)[1],\
                          c=Final_Assgn_Cluster)
    axs[i//3,i%3].scatter(np.transpose(Final_Cluster_Centroids)[0],\
                          np.transpose(Final_Cluster_Centroids)[1],\
                              c=np.arange(i+2),marker='x')
    metric[i] = Cluster_Metric
plt.figure()
plt.plot(K,metric)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Metric")
plt.title("K-means of Multi-Blob Data")
#"""

############################################################################

# K-means of Moon Data
#"""
n_samples = 1000
X, y = noisy_moons = make_moons(n_samples=n_samples, noise= .1)
display_cluster(X)

fig, axs = plt.subplots(3,3,figsize=(25, 25))
metric = np.zeros(9,'d')
K = np.array([2,3,4,5,6,7,8,9,10])
for i in range(9):
    [ Final_Cluster_Distance , Cluster_Metric , Final_Cluster_Centroids ,\
     Final_Assgn_Cluster ] = GUC_Kmean(X, i+2, distance_type)
    #plt.figure()
    axs[i//3,i%3].scatter(np.transpose(X)[0],np.transpose(X)[1],c=Final_Assgn_Cluster)
    axs[i//3,i%3].scatter(np.transpose(Final_Cluster_Centroids)[0],\
                np.transpose(Final_Cluster_Centroids)[1],c=np.arange(i+2),\
                    marker='x')
    metric[i] = Cluster_Metric
plt.figure()
plt.plot(K,metric)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Metric")
plt.title("K-means of Moon Data")
#"""

############################################################################

#"""
customer_data = pd.read_csv("Customer data.csv")
customer_data.drop_duplicates(inplace = True)
customer_data.dropna(inplace = True)
customer_data.set_index(['ID'],inplace = True)
customer_data.info()
#print(customer_data.columns)
#z = customer_data[['Age']]
# y = customer_data.loc[100000001]
# print(y)
# x = customer_data.iloc[0]
# print(x)

#X = customer_data.to_numpy()
scale= StandardScaler()
customer_data = scale.fit_transform(customer_data)
X = customer_data
metric = np.zeros(9,'d')
K = np.array([2,3,4,5,6,7,8,9,10])
for i in range(9):
    [ Final_Cluster_Distance , Cluster_Metric , Final_Cluster_Centroids ,\
     Final_Assgn_Cluster ] = GUC_Kmean(X, i+2, distance_type)
    metric[i] = Cluster_Metric
plt.figure()
plt.plot(K,metric)
plt.xlabel("Number of Clusters, K")
plt.ylabel("Metric")
plt.title("K-means of Customer Data")
#"""