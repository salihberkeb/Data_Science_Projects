import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps=PorterStemmer()
data=pd.read_csv("Emotion_classify_Data.csv")
word_lists=[]
for i in range(0,len(data)):
    yorum=re.sub('a-zA-Z',' ',data["Comment"][i])
    yorum=yorum.split()
    yorum=[ps.stem(kelime,to_lowercase=True) for kelime in yorum if kelime not in set(stopwords.words('english',ignore_lines_startswith="not"))]
    yorum=' '.join(yorum)
    word_lists.append(yorum)

# a=nltk.download('stopwords')


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
x=cv.fit_transform(word_lists).toarray()
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)



from sklearn.metrics import confusion_matrix,classification_report

# =============================================================================
# LOGISTIC REG accuracy = 94%
# =============================================================================
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train)

y_pred = logr.predict(x_test)
# print(y_pred)
# print(y_test)


print("LOGISTIC REG")
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test, y_pred))

# =============================================================================
# KNN accuracy = 70%
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print("KNN")
print(cm)
print(classification_report(y_test, y_pred))
# =============================================================================
# SVC accuracy = 92%
# =============================================================================

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)
print(classification_report(y_test, y_pred))
# =============================================================================
# NAİVE BAYES accuracy = 65%
# =============================================================================

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB naive_bayes')
print(cm)
print(classification_report(y_test, y_pred))

# =============================================================================
# DECİSİON TREE accuracy = 93%
# =============================================================================

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)
print(classification_report(y_test, y_pred))

# =============================================================================
# RANDOM FOREST accuracy = 93%
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)
print(classification_report(y_test, y_pred))
# TAHMİN ORANLARINI VERİR
y_proba = rfc.predict_proba(x_test)

# print(y_test)
print(y_proba[:,0])
    
# =============================================================================
# 7. ROC , TPR, FPR değerleri bu datasett için 3 boyutlu grafik gerekir
# =============================================================================
"""

from sklearn import metrics

fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='anger')
print("FPR")
print(fpr)
print("TPR")
print(tpr)
# print("thresholds")
# print(thold)


plt.plot(fpr, tpr) 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate") 
plt.title("ROC Curve") 
plt.show()
"""

