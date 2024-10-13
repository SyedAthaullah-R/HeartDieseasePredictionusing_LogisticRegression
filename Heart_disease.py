import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
heart_data = pd.read_csv(r"C:\Machine_learning_Projects\Heart_disease\heart_disease_data.csv")
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)
input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)


# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

st.markdown("<h1 style='text-align: center; font-size:35px;'>HEART DISEASE PREDICTION</h1>",
 unsafe_allow_html=True,)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.text("            by Syed Athaullah")  
age = st.number_input("Age",5,99)
sex = st.number_input("Sex (0 = female, 1 = male)",0,1)
cp = st.number_input("Chest pain type",0,3)
trestbps = st.number_input("Resting blood pressure (in mm Hg)",45,300)
chol = st.number_input("Serum cholestoral in mg/dl",10,450)
fbs = st.number_input("Fasting blood sugar > 120 mg/dl",0,1)
restecg = st.number_input("Resting electrocardiographic results",0,1)
thalach = st.number_input("Maximum heart rate achieved",100,200)
exang = st.number_input("Exercise induced angina (1 = yes; 0 = no)",0,1)
oldpeak = st.number_input("ST depression induced by exercise relative to rest",0,5)
slope = st.number_input("Slope of the peak exercise ST segment",0,2)
ca = st.number_input("Number of major vessels (0-3) colored by flourosopy",0,3)
thal = st.number_input("Thal",1,3)
input_data = (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
if st.button("Predict Heart Hisease with solution"):
    input_data = (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    if (prediction[0]== 0):
       a='The person does not have heart disease'
       st.success(a)
    else:
      b='The person has Heart Disease.\n Please visit the nearby Doctor immediately'
      st.success(b)



