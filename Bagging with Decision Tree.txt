# Bagging with Decision Tree
# Import Libraries

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Synthetics Data

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    random_state= 20)
print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.2,random_state=0)

#Initialize the Decision Tree Classifier
Base_estimator = DecisionTreeClassifier(random_state=20)

#Initialize the bagging classifier
bagging_model = BaggingClassifier(
estimator = Base_estimator,
n_estimators = 15,
random_state = 70)

# Training the bagging classifier
bagging_model.fit(X_train,y_train)

# Evaluate Model
y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
