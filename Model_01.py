import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df = pd.read_csv("/Users/mr.bajrangi/Library/Mobile Documents/com~apple~CloudDocs/Visual Studio Code/File/DS/user_behavior_dataset.csv")
#df[""] = le.fit_transform(df[""])

X = df[["User ID","Device Model","Operating System","App Usage Time (min/day)","Battery Drain (mAh/day)","Number of Apps Installed","Data Usage (MB/day)","Age","Gender","User Behavior Class"]].values
Y = df[["Screen On Time (hours/day)"]].values

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

print(Y_pred)