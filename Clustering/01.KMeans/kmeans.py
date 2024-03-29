import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("musteriler.csv")

x=veriler.iloc[:,3:].values
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=3,init='k-means++')
kmeans.fit(x)

print(kmeans.cluster_centers_)
sonuclar=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)