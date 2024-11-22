import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 

file_path = '/Users/mr.bajrangi/Library/Mobile Documents/com~apple~CloudDocs/Visual Studio Code/File/DS/diabetes.csv'
diabetes_df = pd.read_csv(file_path)

a = diabetes_df.head()

b = diabetes_df.describe()
print(a)
print(b)

columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for column in columns_with_zeros:
    median_value = diabetes_df[column].median()
    diabetes_df[column] = diabetes_df[column].replace(0, median_value)

c = diabetes_df[columns_with_zeros].describe()
print(c)


X = diabetes_df.drop('Outcome', axis=1) 
y = diabetes_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Adjust test_size and random_state as needed

decision_tree = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=4)

v = decision_tree.fit(X_train, y_train)
print(v)