#Bagging with Decision Tree
#import libraries

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x , y = make_classification(
    n_samples=1000,
    n_features=10,
    random_state=20,
    )

print(x,y)

x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2,random_state=0)

#Initialize the Decision Tree Classifier
Base_estimator = DecisionTreeClassifier(random_state=20)

#Initialize the Bagging classifier

bagging_model = BaggingClassifier(
    estimator=Base_estimator,
    n_estimators=15,
    random_state=20
    )
bagging_model.fit(x_train,y_train)

y_pred = bagging_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#_____________________________________________

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models (learners)
base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier())
]

# Define the meta-model (stacking model)
meta_model = LogisticRegression()

# Create the stacking classifier
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)

# Train the stacking classifier
stacking_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = stacking_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Stacking Classifier: {accuracy * 100:.3f}%')