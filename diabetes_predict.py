import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes_data = pd.read_csv('diabetes.csv')  # Replace 'diabetes.csv' with your dataset filename

# Split the data into features (X) and target variable (y)
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of models to evaluate
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, solver='liblinear')),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC(random_state=42))
]

# Train and evaluate each model
for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy}")

#OUTPUT
#Logistic Regression Accuracy: 0.7597402597402597
#Decision Tree Accuracy: 0.7532467532467533
#Random Forest Accuracy: 0.7467532467532467
#SVM Accuracy: 0.7662337662337663
