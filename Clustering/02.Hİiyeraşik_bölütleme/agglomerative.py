import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("musteriler.csv")

x=veriler.iloc[:,3:].values

# =============================================================================
# KMEANS
# =============================================================================
from sklearn.cluster import KMeans
# kmeans=KMeans(n_clusters=3,init='k-means++')
# kmeans.fit(x)
# # print(kmeans.cluster_centers_)
# sonuclar=[]

# for i in range(1,11):
#     kmeans=KMeans(n_clusters=i,init='k-means++',random_state=123)
#     kmeans.fit(x)
#     sonuclar.append(kmeans.inertia_)

# plt.plot(range(1,11),sonuclar)

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=123)
y_tahmin1=kmeans.fit_predict(x)

plt.scatter(x[y_tahmin1==0,0],x[y_tahmin1==0,1],s=100,c='red')
plt.scatter(x[y_tahmin1==1,0],x[y_tahmin1==1,1],s=100,c='blue')
plt.scatter(x[y_tahmin1==2,0],x[y_tahmin1==2,1],s=100,c='green')
plt.scatter(x[y_tahmin1==3,0],x[y_tahmin1==3,1],s=100,c='brown')
plt.title("KMenas")
plt.show()





# =============================================================================
# HC
# =============================================================================

from sklearn.cluster import AgglomerativeClustering

ac=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
y_tahmin=ac.fit_predict(x)
print(y_tahmin)
plt.scatter(x[y_tahmin==0,0],x[y_tahmin==0,1],s=100,c='red')
plt.scatter(x[y_tahmin==1,0],x[y_tahmin==1,1],s=100,c='blue')
plt.scatter(x[y_tahmin==2,0],x[y_tahmin==2,1],s=100,c='green')
plt.scatter(x[y_tahmin==3,0],x[y_tahmin==3,1],s=100,c='brown')
plt.title("Hiyerasik")
plt.show()

import scipy.cluster.hierarchy as sch

dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.show()









