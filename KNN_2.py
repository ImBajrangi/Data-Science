import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

a=pd.read_excel("/Users/mr.bajrangi/File/DS/Fruit Classification.xlsx")
x=a[[ 'weight(grams)' , 'Size' ]].values
y=a[ 'fruit type' ].values
print(x)
print(y)

knn = KNeighborsClassifier(n_neighbors=3)

le = preprocessing.LabelEncoder()
a['Size'] = le.fit_transform(a['Size'])

le = preprocessing.LabelEncoder()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# training,testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
new = np.array([[140,7,0]])
y_pred = knn.predict(new)
print(y_pred)
