import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps=PorterStemmer()
data=pd.read_csv("train.csv",header=None,sep=";")
word_lists=[]
for i in range(0,len(data)):
    yorum=re.sub('a-zA-Z',' ',data[0][i])
    yorum=yorum.split()
    yorum=[ps.stem(kelime,to_lowercase=True) for kelime in yorum if kelime not in set(stopwords.words('english',ignore_lines_startswith="not"))]
    yorum=' '.join(yorum)
    word_lists.append(yorum)

# a=nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000)
x=cv.fit_transform(word_lists).toarray()
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train)

y_pred = logr.predict(x_test)

while 1:
    print("\n")
    user_input=str(input("talk: "))
    if user_input=="q":
        break
    else:
        user_input=user_input.split()
        user_input=[ps.stem(kelime,to_lowercase=True) for kelime in user_input if kelime not in set(stopwords.words('english'))]
        user_input=' '.join(user_input)
        x_user=cv.transform([user_input]).toarray()
        y_pred_user=logr.predict(x_user)
        print(y_pred_user)

