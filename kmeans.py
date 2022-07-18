# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('2008.csv')
# dataset1=pd.read_csv('2008.csv')
X = dataset.iloc[:, [0,1,2]].values
# print(len(X))
# X1=dataset1.iloc[:,[0,1,2]].values
# output=findoverlap(X1,X)
# print(len(output))
teX= X[:,2].reshape(len(X),1)
# dataset=np.load('output_overlap.npy')
# X = dataset[:, [2,5]]
# X1=dataset[:, [3,4,5]]
# teX=X
# teX.tolist()
# print(len(teX))
# teX=[x for x in teX if x[0]!=0 and x[1]!=0]
# teX=np.array(teX)
# print(len(teX))

# teX1= X1[:,2].reshape(len(X1),1)
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
teX = sc_X.fit_transform(teX)



# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(teX)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, init = 'k-means++')
y_kmeans = kmeans.fit_predict(teX)
# mean0=np.mean(X[y_kmeans == 0, 2])
# mean1=np.mean(X[y_kmeans == 1, 2])
# mean2=np.mean(X[y_kmeans == 2, 2])
# mean3=np.mean(X[y_kmeans == 3, 2])
# len0=len(X[y_kmeans == 0, 2])
# len1=len(X[y_kmeans == 1, 2])
# len2=len(X[y_kmeans == 2, 2])
# len3=len(X[y_kmeans == 3, 2])
# print('mean yield and numbers of cluster1',mean0,len0)
# print('mean yield and numbers of cluster2',mean1,len1)
# print('mean yield and numbers of cluster3',mean2,len2)
# print('mean yield and numbers of cluster4',mean3,len3)
# print('total yield:',mean0*len0+mean1*len1+mean2*len2+mean3*len3)


# mean0=np.mean(X1[y_kmeans == 0, 2])
# mean1=np.mean(X1[y_kmeans == 1, 2])
# mean2=np.mean(X1[y_kmeans == 2, 2])
# mean3=np.mean(X1[y_kmeans == 3, 2])
# gap01=mean0-mean1
# gap02=mean0-mean2
# gap03=mean0-mean3
# gap12=mean1-mean2
# gap13=mean1-mean3
# gap23=mean2-mean3
# len0=len(X1[y_kmeans == 0, 2])
# len1=len(X1[y_kmeans == 1, 2])
# len2=len(X1[y_kmeans == 2, 2])
# len3=len(X1[y_kmeans == 3, 2])
# print('mean yield and numbers of cluster1',mean0,len0)
# print('mean yield and numbers of cluster2',mean1,len1)
# print('mean yield and numbers of cluster3',mean2,len2)
# print('mean yield and numbers of cluster4',mean3,len3)
# print('total yield:',mean0*len0+mean1*len1+mean2*len2+mean3*len3)
# testp=4.5
# print('According to given corpflow:',testp,'The evaluation of total yield is:',)

# Visualising the clusters

# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 1, c = 'red', label = str(mean0))
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 1, c = 'blue', label = str(mean1))
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 1, c = 'green', label = str(mean2))
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 1, c = 'cyan', label = str(mean3))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 1, c = 'blue', label = 'c1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 1, c = 'blue', label = 'c2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 1, c = 'blue', label = 'c3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 1, c = 'blue', label = 'c4')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 10, c = 'magenta', label = 'Cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('Clusters of cropflow in 2010 and 2008(wheat)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
plt.xlabel('2010')
plt.ylabel('2008')
plt.legend()
plt.show()

# plt.scatter(X1[y_kmeans == 0, 0], X1[y_kmeans == 0, 1], s = 1, c = 'red', label = str(mean0))
# plt.scatter(X1[y_kmeans == 1, 0], X1[y_kmeans == 1, 1], s = 1, c = 'blue', label = str(mean1))
# plt.scatter(X1[y_kmeans == 2, 0], X1[y_kmeans == 2, 1], s = 1, c = 'green', label = str(mean2))
# plt.scatter(X1[y_kmeans == 3, 0], X1[y_kmeans == 3, 1], s = 1, c = 'cyan', label = str(mean3))
# # plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 10, c = 'magenta', label = 'Cluster 5')
# # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
# plt.title('Clusters of cropflow in 2008(wheat)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend()
# plt.show()

