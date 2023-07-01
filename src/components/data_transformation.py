from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys,os
from dataclasses import dataclass

from src.utils import save_object

@dataclass
class DataTransformationConfig :
    preprocessor_obj_file = os.path.join('artifacts','preprocessor.pkl')


# Data transformation class
class DataTransformation :

    def __init__(self):
        self.data_transs_config = DataTransformationConfig()

    def get_data_transformation_object(self):

        try :
            logging.info('Data transformation initiated')

            logging.info('Data Transformation pipeline initiated')
            columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='mean')),
                    ('scaler',StandardScaler())

                ]

            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,columns)
            ])

            logging.info('Data transformation completed')

            return preprocessor


        except Exception as e:
            logging.info('Exception at data transformation stage')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data_path, test_data_path):
        logging.info('Data transformation initiated')
        try :
            # load train and test data
            train_df = pd.read_csv(train_data_path)
            test_df  = pd.read_csv(test_data_path)
            logging.info('Read train and test data completed')
            logging.info(f'Train data head:\n {train_df.head().to_string}')
            logging.info(f'Test data head:\n {test_df.head().to_string}')

            logging.info('Obtaining preprocessor object')

            preprocessing_obj = self.get_data_transformation_object()
            
            target_column = 'Outcome'
            drop_columns=[target_column]

            # dividing dataset into dependent and independent dataset
            # Training data
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            output_feature_train_df = train_df[target_column]

            #test data
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            output_feature_test_df = test_df[target_column]

            # Data transformation

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(output_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(output_feature_test_df)]

            #save the pre processor object
            save_object(
                file_path=self.data_transs_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )

            logging.info('Applying preprocessor object on training and test data sets')

            return (
                train_arr,
                test_arr,
                self.data_transs_config.preprocessor_obj_file
            )
            

        except Exception as e :
            logging.info('Exception occured during data transformation')
            raise CustomException(e,sys)