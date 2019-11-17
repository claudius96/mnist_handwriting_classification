import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

#importing the dataset
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
# Split the data into training and test sets
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X =sc.fit_transform(X)

#applying principal component analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(X)




wcss = []
from sklearn.cluster import KMeans
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',
                    n_init=10,max_iter=300)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
components = pca.components_
plt.plot(range(1, 11,1),wcss)
plt.title('Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300)
y_pred =kmeans.fit_predict(X)

colors = ['green', 'blue', 'red', 'cyan']
for i in range(4):
    plt.scatter(X[y_pred==i,0], X[y_pred==i,1],c=colors[i],s=70)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            c='yellow',s=300,label='centroids',marker='X')
plt.title('Using the kmeans algorithm')
plt.show()









    
    
    
    
    
    
    
    
    


