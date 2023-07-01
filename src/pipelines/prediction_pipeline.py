import os,sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd
import numpy as np
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig

class PredictPipeline:

    def __init__(self):
        pass

    def predict(self,features):
        try:
            #get the file path of preprocessor object and trained model
            preprocessor_config = DataTransformationConfig()
            model_config = ModelTrainerConfig()
            # load preprocessor object and trained model
            preprocessor= load_object(preprocessor_config.preprocessor_obj_file)
            model = load_object(model_config.trained_model_file_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred


        except Exception as e:
            logging.info("Error occured during prediction")
            raise CustomException(e,sys)

class CustomData:

    def __init__(self,
                 pregnancies:int,
                 glucose:float,
                 bloodPressure:float,
                 skinThickness:float,
                 insulin:float,
                 bmi:float,
                 diabetesPedigreeFunction:float,
                 age:int) :
        
            self.pregnancies = pregnancies
            self.glucose = glucose
            self.bloodPressure = bloodPressure
            self.skinThickness = skinThickness
            self.insulin = insulin
            self.bmi = bmi
            self.diabetesPedigreeFunction = diabetesPedigreeFunction
            self.age = age


    def get_data_as_dataframe(self):
        input_dct = {
               'Pregnancies': [self.pregnancies],
                'Glucose': [self.glucose], 
                'BloodPressure':[self.bloodPressure], 
                'SkinThickness' : [self.skinThickness], 
                'Insulin': [self.insulin],
                'BMI':[self.bmi], 
                'DiabetesPedigreeFunction': [self.diabetesPedigreeFunction], 
                'Age' : [self.age]
            }

        df = pd.DataFrame.from_dict(input_dct)
        return df
                