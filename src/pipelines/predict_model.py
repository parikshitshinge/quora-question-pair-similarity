import sys 
from pathlib import Path
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = "./models/model.pkl"            
            vectorizer_path = "./models/vectorizer.pkl"
            model = load_object(file_path=model_path)
            logging.info("model.pkl is loaded")
            vectorizer = load_object(file_path=vectorizer_path)
            logging.info("vectorizer.pkl is loaded")
            vectorized_data = vectorizer.transform(features)
            logging.info("Query point is vectorized")
            pred = model.predict(vectorized_data)
            logging.info("Successfully predicted")
            return pred
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, question1:str, question2:str):
        self.question1 = question1
        self.question2 = question2 
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                            "question1" : [self.question1],
                            "question2" : [self.question2]
            }
            logging.info("Returning query point as DataFrame")
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            CustomException(e, sys)