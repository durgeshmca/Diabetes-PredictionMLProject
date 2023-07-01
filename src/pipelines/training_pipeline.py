import os,sys
import pandas
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    ## Data ingestion
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initialize_ingestion()
    print( train_data_path,test_data_path)

    data_transformation = DataTransformation()

    train_arr,test_arr,preprocessor_obj_file= data_transformation.initiate_data_transformation(
                                                train_data_path=train_data_path,
                                                test_data_path=test_data_path
                                                )
    
    print(preprocessor_obj_file)
    # ## train model
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)