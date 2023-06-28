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

##
Diabetes ML Prediction GUI
##


This project aims to predict the risk of diabetes using machine learning algorithms. It provides a graphical user interface (GUI) for users to input their health information and obtain a prediction of their diabetes risk.

## Dataset

The project utilizes the `diabetes.csv` dataset, which contains various health features such as glucose levels, BMI, blood pressure, and more. The dataset is included in the repository.

## Dependencies

To run the project, you need to have the following dependencies installed:

- Python 3
- pandas
- scikit-learn
- tkinter

You can install the required dependencies using pip:
``` pip install pandas scikit-learn tkinter ```


## Usage

1. Clone the repository:

```git clone https://github.com/your-username/diabetes-ml-prediction.git```

2. Change into the project directory:
```cd diabetes-ml-prediction```

3. Run the GUI script:
   
```python diabetes_pred_gui.py```

5. The GUI window will open. Enter the required health information, such as glucose levels, BMI, age, etc.

6. Click the "Predict" button to obtain the diabetes risk prediction.

## Feature Importance

The model used in this project considers the following features in predicting diabetes risk, in order of importance:

- Glucose
- BMI
- Age
- Diabetes Pedigree Function
- Blood Pressure
- Pregnancies
- Insulin
- Skin Thickness

Please provide accurate information for these features to improve the accuracy of the prediction.

## Contributions

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.


