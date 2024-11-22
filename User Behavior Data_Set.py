# Logistic Regression

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline


df = pd.read_csv('user_behavior_dataset.csv')
# print(df)
le = preprocessing.LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Device Model'] = le.fit_transform(df['Device Model'])
df['Operating System'] = le.fit_transform(df['Operating System'])
# print(df)

x = df[['Gender','Device Model','Operating System','App Usage Time (min/day)','Screen On Time (hours/day)','Battery Drain (mAh/day)','Number of Apps Installed','Data Usage (MB/day)','Age']].values
y = df['User Behavior Class'].values

# # training,testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
Logreg = LogisticRegression()
# # fit the model with data
Logreg.fit(x_train,y_train)
print(x_train)
print(y_train)

# prediction
y_pred=Logreg.predict(x_test)
print(y_pred)
print(y_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)


accuracy=accuracy_score(y_test,y_pred)
print("accuracy",accuracy)

precision = precision_score(y_test,y_pred, average='macro')
print("precision",precision)

recall = recall_score(y_test,y_pred, average='macro')
print("recall = ",recall)

f1 = f1_score(y_test,y_pred, average='macro')
print("f1",f1)

class_names= [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
print
# create headmap
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="Greens",fmt="d",annot_kws={"size":10})
ax.xaxis.set_label_position("bottom")
plt.tight_layout()
plt.title("confusion matrix", y=1.4)
plt.xlabel('actual label')
plt.xlabel('predicated labels')
#plt.show
