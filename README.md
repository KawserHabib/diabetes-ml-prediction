# Diabetes Prediction using Machine Learning

This project aims to predict the risk of diabetes using machine learning techniques. It utilizes a dataset containing various features related to diabetes, such as glucose levels, BMI, age, and blood pressure. By training different machine learning models on this dataset, we can predict whether an individual is likely to have diabetes or not.

## Dataset

The dataset used in this project is stored in the file `diabetes.csv`. It contains the following columns:

- `Glucose`: Plasma glucose concentration.
- `BMI`: Body mass index.
- `Age`: Age of the individual.
- `DiabetesPedigreeFunction`: Diabetes pedigree function.
- `BloodPressure`: Blood pressure levels.
- `Pregnancies`: Number of times pregnant.
- `Insulin`: Insulin levels.
- `SkinThickness`: Skin thickness.

The target variable is `Outcome`, which indicates whether an individual has diabetes (1) or not (0).


############

This script will output the accuracy of each model on the test set.

## Machine Learning Models

The following machine learning models are utilized in this project:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Each model is trained on the training set and evaluated on the test set to measure its prediction accuracy.

## Results

The accuracy of each model on the test set is as follows:

- Logistic Regression: 0.75
- Decision Tree: 0.72
- Random Forest: 0.77
- SVM: 0.76

Please note that these results are specific to the given dataset and may vary with different datasets or model configurations.

