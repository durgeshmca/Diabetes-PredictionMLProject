from flask import Flask,render_template,request,jsonify
import sys,os
import pickle
from src.exception import CustomException
import pandas as pd
from src.utils import load_object
from src.pipelines.prediction_pipeline import PredictPipeline,CustomData

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
            glucose = float(request.form.get('glucose'))
            bloodPressure = float(request.form.get('bloodPressure'))
            skinThickness = float(request.form.get('skinThickness'))
            insulin = float(request.form.get('insulin'))
            bmi = float(request.form.get('bmi'))
            diabetesPedigreeFunction = float(request.form.get('diabetesPedigreeFunction'))
            age = int(request.form.get('age'))
            data = CustomData(
                pregnancies,
                glucose,
                bloodPressure,
                skinThickness,
                insulin,
                bmi,
                diabetesPedigreeFunction,
                age
            )

            

            input_df = data.get_data_as_dataframe()
            pipeline = PredictPipeline()
            output = pipeline.predict(input_df)

   

            # scaled_input = standard_scaler.transform(input_df)
            # output = model.predict(scaled_input)

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