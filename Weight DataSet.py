import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

new=pd.read_excel('/content/dSfile.xlsx')

X=new[['weight(x)','height(y)']].values
Y=new['Class'].values

#training,testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#KNN
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,Y_train)

new=np.array([[57,170]])

Y_pred=knn.predict(new)

print(Y_pred)
print(new)
