import os
import sys

from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## initialize the data ingestion


@dataclass  #used to create class variables no need to start with init 
class DataIngestionConfig:
    # set the configuration of data sources
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str  = os.path.join('artifacts','test.csv')
    raw_data_path:str   = os.path.join('artifacts','raw.csv')

## Create the data ingestion class
class DataIngestion :
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initialize_ingestion(self):
        logging.info("Data Ingestion starts")

        try:
            # here you can get data from any source from MongoDb , Mysql, file etc
            df = pd.read_csv(os.path.join('notebooks','data','diabetes_updated'))
            logging.info('Data set read as pandas dataframe')

            # now create raw data path if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            #save data to raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw Data saved')
            #get train and test data using train_test_split
            train_set,test_set = train_test_split(df,test_size=0.25,random_state=42)

            #now save train and test data

            train_set.to_csv(self.ingestion_config.train_data_path,index=True,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=True,header=True)
            logging.info('Train and Test Set Data created.Data Ingestion completed')
            print(self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )





        except Exception as e:
            logging.info("Exception occured at Data ingestion stage")
            raise CustomException(e,sys)
        
        

