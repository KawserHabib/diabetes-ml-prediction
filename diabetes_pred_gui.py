import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, Label, Entry, Button, messagebox
from tkinter import Text


# Load the dataset
diabetes_data = pd.read_csv('diabetes.csv')  # Replace 'diabetes.csv' with your dataset filename

# Split the data into features (X) and target variable (y)
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the imputer with the training data
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)

# Preprocess the data
scaler = StandardScaler()
#X_train_scaled = pd.DataFrame(scaler.fit_transform(imputer.transform(X_train)), columns=X_train.columns)

X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)


# Train the model (using SVC as an example)
model = SVC(probability=True,random_state=42)
model.fit(X_train_scaled, y_train)

# Create the GUI window
window = Tk()
window.title('Diabetes Prediction')

# Create labels and entry fields for input features
pregnancies_label = Label(window, text='Pregnancies:')
pregnancies_label.grid(row=0, column=0)
pregnancies_entry = Entry(window)
pregnancies_entry.grid(row=0, column=1)

glucose_label = Label(window, text='Glucose:')
glucose_label.grid(row=1, column=0)
glucose_entry = Entry(window)
glucose_entry.grid(row=1, column=1)

blood_pressure_label = Label(window, text='Blood Pressure:')
blood_pressure_label.grid(row=2, column=0)
blood_pressure_entry = Entry(window)
blood_pressure_entry.grid(row=2, column=1)

skin_thickness_label = Label(window, text='Skin Thickness:')
skin_thickness_label.grid(row=3, column=0)
skin_thickness_entry = Entry(window)
skin_thickness_entry.grid(row=3, column=1)

insulin_label = Label(window, text='Insulin:')
insulin_label.grid(row=4, column=0)
insulin_entry = Entry(window)
insulin_entry.grid(row=4, column=1)

bmi_label = Label(window, text='BMI:')
bmi_label.grid(row=5, column=0)
bmi_entry = Entry(window)
bmi_entry.grid(row=5, column=1)

diabetes_pedigree_label = Label(window, text='Diabetes Pedigree:')
diabetes_pedigree_label.grid(row=6, column=0)
diabetes_pedigree_entry = Entry(window)
diabetes_pedigree_entry.grid(row=6, column=1)

age_label = Label(window, text='Age:')
age_label.grid(row=7, column=0)
age_entry = Entry(window)
age_entry.grid(row=7, column=1)

# Define the predict_diabetes function
# Create a label to display feature importance
feature_importance_label = Label(window, text="Feature Importance")
feature_importance_label.grid(row=9, column=0, columnspan=2)

# Create a text area to show feature importance values
feature_importance_text = Text(window, height=8, width=40)
feature_importance_text.grid(row=10, column=0, columnspan=2)
feature_importance_text.insert(END, "Feature Importance\n")
feature_importance_text.insert(END, "---------------------------\n")
feature_importance_text.insert(END, f"Glucose: {feature_importance['Glucose']}\n")
feature_importance_text.insert(END, f"BMI: {feature_importance['BMI']}\n")
feature_importance_text.insert(END, f"Age: {feature_importance['Age']}\n")
feature_importance_text.insert(END, f"Diabetes Pedigree: {feature_importance['DiabetesPedigreeFunction']}\n")
feature_importance_text.insert(END, f"Blood Pressure: {feature_importance['BloodPressure']}\n")
feature_importance_text.insert(END, f"Pregnancies: {feature_importance['Pregnancies']}\n")
feature_importance_text.insert(END, f"Insulin: {feature_importance['Insulin']}\n")
feature_importance_text.insert(END, f"Skin Thickness: {feature_importance['SkinThickness']}\n")

# Define the predict_diabetes function
def predict_diabetes():
    # Get the input feature values from the GUI
    pregnancies = pregnancies_entry.get()
    glucose = glucose_entry.get()
    blood_pressure = blood_pressure_entry.get()
    skin_thickness = skin_thickness_entry.get()
    insulin = insulin_entry.get()
    bmi = bmi_entry.get()
    diabetes_pedigree = diabetes_pedigree_entry.get()
    age = age_entry.get()

    # Set default values for empty fields
    pregnancies = int(pregnancies) if pregnancies else 0
    glucose = float(glucose) if glucose else 0.0
    blood_pressure = float(blood_pressure) if blood_pressure else 0.0
    skin_thickness = float(skin_thickness) if skin_thickness else 0.0
    insulin = float(insulin) if insulin else 0.0
    bmi = float(bmi) if bmi else 0.0
    diabetes_pedigree = float(diabetes_pedigree) if diabetes_pedigree else 0.0
    age = int(age) if age else 0

    # Create a DataFrame with the input features
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Handle missing values in the input data
    input_data_imputed = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)

    # Preprocess the input features
    input_data_scaled = pd.DataFrame(scaler.transform(input_data_imputed), columns=input_data_imputed.columns)

    # Make the prediction
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1] * 100

    # Show the prediction result
    messagebox.showinfo('Diabetes Prediction', f'The predicted outcome is {prediction[0]}. \n\nDiabetes risk probability: {probability:.2f}%')

# Create the predict button
predict_button = Button(window, text='Predict', command=predict_diabetes)
predict_button.grid(row=8, column=0, columnspan=2)



# Run the GUI main loop
window.mainloop()
