import os 
import sys 
from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object, evaluate_models

import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
# Metrics
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

@dataclass 
class TrainModelConfig:
    trained_model_file_path = os.path.join("models", "model.pkl")
    
class TrainModel:
    def __init__(self):
        self.train_model_config = TrainModelConfig()
        
    def initiate_train_model(self, X_train, X_test, y_train, y_test,):
        try:
            
            models = {
                "Logistic Regression":LogisticRegression(solver='lbfgs', max_iter=3000),
                "BernoulliNM":BernoulliNB(),
                "SVC":SVC(),
                "Gradient Boosting Classifier":GradientBoostingClassifier(),
                "XG Boost":XGBClassifier()
            }
            
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models )
            
            # Get the best model score & name from dict 
            best_model_score = max(sorted(model_report.values()))            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score>0.89532403134099: # This is we got from random model
                raise CustomException("No best model found")
            
            save_object(file_path = self.train_model_config.trained_model_file_path, obj=best_model)
            
            y_pred = best_model.predict(X_test)
            
            lg_loss = log_loss(y_test, y_pred)
            
            return lg_loss
            
        except Exception as e:
            raise CustomException(e, sys)