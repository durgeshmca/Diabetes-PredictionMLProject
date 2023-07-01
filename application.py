from flask import Flask,render_template,request,jsonify
import sys,os
import pickle
from src.exception import CustomException
import pandas as pd
from src.utils import load_object

application = Flask(__name__)
app = application

standard_scaler = load_object("artifacts/standard_scaler.pkl")
model = load_object('artifacts/model.pkl')

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    try :
        if request.method == "POST":
            pregnancies = int(request.form.get('pregnancies'))
            glucose = int(request.form.get('glucose'))
            bloodPressure = int(request.form.get('bloodPressure'))
            skinThickness = float(request.form.get('skinThickness'))
            insulin = int(request.form.get('insulin'))
            bmi = float(request.form.get('bmi'))
            diabetesPedigreeFunction = float(request.form.get('diabetesPedigreeFunction'))
            age = int(request.form.get('age'))

            input_dct = {
               'Pregnancies': [pregnancies],
                'Glucose': [glucose], 
                'BloodPressure':[bloodPressure], 
                'SkinThickness' : [skinThickness], 
                'Insulin': [insulin],
                'BMI':[bmi], 
                'DiabetesPedigreeFunction': [diabetesPedigreeFunction], 
                'Age' : [age]
            }

            input_df = pd.DataFrame.from_dict(input_dct)

   

            scaled_input = standard_scaler.transform(input_df)
            output = model.predict(scaled_input)

            result = "Person is not diabetic"
            if output[0] == 1:
                result = "Person is diabetic"
            
            
            return render_template('prediction_form.html',final_result=result)
        else :
            return render_template('prediction_form.html')
    except Exception as e:
        raise(e)

    

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)