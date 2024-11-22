import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Create a k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


Logreg = LogisticRegression()
# fit the model with data
Logreg.fit(X_train,y_train)
# prediction
y_pred=Logreg.predict(X_test)
print(y_pred)
print(y_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
