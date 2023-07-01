import os,sys
import numpy as np
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

##
# Saves objects into pickle file
##
def save_object(file_path,obj):

    try :
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path, 'wb') as f :
            pickle.dump(obj,f)

    except Exception as e:
        logging.info(f'Exception occured during saving object at {file_path}')
        raise CustomException(e,sys)
    
##
# Evaluates the models
##
    
def evaluate_models(X_train,y_train,X_test,y_test,models:dict):
    try :
        logging.info(f"Evaluating model: {models}")
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            #train model
            #print(len(X_train),len(y_train))
            model.fit(X_train,y_train)

            #predict testing data
            y_test_predict = model.predict(X_test)

            #get r2 score for train and test data
            train_model_score = r2_score(y_test,y_test_predict)
            #add to report
            report[list(models.keys())[i]] = train_model_score
            
        return report


    except Exception as e:
         logging.info('Exception occured during model training')
         raise CustomException(e,sys)
    
##
# Load object from pickle file
##
def load_object(file_path):

    try :
        with open(file_path,'rb') as f:
            obj = pickle.load(f)
            return obj

    except Exception as e:
        logging.info(f'Exception occured during loading object at {file_path}')
        raise CustomException(e,sys)
    

    ###
    # Hypertune variable for logical regression model
    ##

def  hypertune_model(X_train,y_train,X_test,y_test,regressor:LogisticRegression):
     try :
        logging.info(f"finding best parameters for logical regressor")
        report = {}
        parameters = {
            'penalty':('l1', 'l2', 'elasticnet'),
            'C':[-3,-2,-1,0,1,2,3],
            'solver': ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'sag')
        }
        regressor = LogisticRegression()
        eval_model = GridSearchCV(regressor,parameters,cv=10,scoring='accuracy')
        eval_model.fit(X_train,y_train)
        report['best_parameters'] = eval_model.best_params_
        report['best_score'] = eval_model.best_score_
        logging.info(f"best parameters are :")
        logging.info(report)
        

        return report
         
     except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

