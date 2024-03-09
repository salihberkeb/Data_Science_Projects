import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("Wine.csv")

X=data.iloc[:,0:13].values
y=data.iloc[:,13].values

from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

# PCA

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)


from sklearn.linear_model import LogisticRegression
# PCA DAN ONCE
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# PCA DAN SONRA
classifier2=LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

y_pred=classifier.predict(X_test)

y_pred2=classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
# PCA OLMADAN ÇIKAN SONUÇ
print("Gerçek/PCAsiz")
cm=confusion_matrix(y_test,y_pred)
print(cm)

# PCA SONRASI ÇIKAN SONUÇ
print("Gerçek/PCA ile")
cm=confusion_matrix(y_test,y_pred2)
print(cm)

print("pcasiz ile pcali")
cm3=confusion_matrix(y_pred,y_pred2)
print(cm3)
