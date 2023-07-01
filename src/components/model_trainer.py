import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,hypertune_model
from dataclasses import dataclass
import sys,os

@dataclass
class ModelTrainerConfig :
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):

        try :
            logging.info("Splitting dependent and independent variables from train and test data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            model_report:dict = hypertune_model(X_train,y_train,X_test,y_test,'LogicalRegression')

            print(f"#############################################\n {model_report} \n##############################\n")
            logging.info(f'Model Report : {model_report}')

            best_model_score = model_report['best_score']
            best_params = model_report['best_parameters']
            
            print(f"\n Hypertuned parameters found {best_params} best score : {best_model_score}\n")
            print("###########################################################################\n")
            # Now train the model with hypertuned parameters
            C = best_params['C']
            penalty = best_params['penalty']
            solver = best_params['solver']

            regressor = LogisticRegression(C=C,penalty=penalty,solver=solver)
            regressor.fit(X_train,y_train)
            # get the performance metrices
            y_pred = regressor.predict(X_test)
            conf_mat = confusion_matrix(y_test,y_pred)
            accuracy = accuracy_score(y_test,y_pred)
            classic = classification_report(y_test,y_pred)

            print('-----------performance metrices-----------')
            print(f'confusion matrix:{conf_mat}')
            print(f'accuracy :{accuracy}')
            print(f'classification report:{classic}')
            logging.info('-----------performance metrices-----------')
            logging.info(f'confusion matrix:{conf_mat}')
            logging.info(f'accuracy :{accuracy}')
            logging.info(f'classification report:{classic}')
            
            # save the model object

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= regressor
            )

        except Exception as e:
            logging.info('Exception occured in model training stage')
            raise CustomException(e,sys)
