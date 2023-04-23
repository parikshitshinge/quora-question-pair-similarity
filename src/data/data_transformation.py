import os
import sys
from datetime import datetime
from src.exception import CustomException
from src.logger import logging
import src.utils
from src.utils import preprocess, feature_engg_basic, compute_tfidf_avg_w2v

from dataclasses import dataclass
import numpy as np
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

# Preprocessing
import string
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize 
#nltk.download('punkt')
from thefuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy



@dataclass
class DataTransformationConfig:
    preprocessed_data_file = os.path.join('./data/processed' , 'processed_data.csv')
    vectorizer_obj_file = os.path.join('models', 'vectorizer.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessed_data(self, raw_data_path):
        """
        This function is responsible for preprocessing data
        """
        try:
           # Read file
            raw_data = pd.read_csv(raw_data_path)
            logging.info('Read raw data completed')
                        
            logging.info('Data transformation started')
            
            processed_data = raw_data.copy()
            
            processed_data['question1'] = processed_data['question1'].fillna("").apply(preprocess)
            logging.info("Preprocessed 'question1'")
            processed_data['question2'] = processed_data['question2'].fillna("").apply(preprocess)
            logging.info("Preprocessed 'question2'")
            
            raw_data_featurized = raw_data.copy()
            raw_data_featurized.fillna("")
            feature_engg_basic(raw_data_featurized)
            logging.info("Basic feature engineering completed")
            
            processed_data_featurized = processed_data.copy()
            feature_engg_adv(processed_data_featurized)
            logging.info("Advanced feature engineering completed")
            
            questions = list(processed_data['question1']) + list(processed_data['question2'])
            questions = [x for x in questions if pd.notnull(x)]
            tfidf = TfidfVectorizer(lowercase=False)
            tfidf.fit_transform(questions)
            
            word2tfidf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
            nlp = spacy.load('en_core_web_lg')
            logging.info('GloVe loaded')
            
            processed_data_tfidfavgw2v = processed_data.copy()
            processed_data_tfidfavgw2v['q1_tfidfavgw2v_features'] = processed_data_tfidfavgw2v['question1'].apply(compute_tfidf_avg_w2v)
            logging.info('Computed TFIDF Avg W2V for question1')
            q1_tfidfavgw2v_features = pd.DataFrame(processed_data_tfidfavgw2v['q1_tfidfavgw2v_features'].values.tolist(), index=processed_data_tfidfavgw2v.index)
            
            processed_data_tfidfavgw2v['q2_tfidfavgw2v_features'] = processed_data_tfidfavgw2v['question2'].apply(compute_tfidf_avg_w2v)
            logging.info('Computed TFIDF Avg W2V for question2')
            q2_tfidfavgw2v_features = pd.DataFrame(processed_data_tfidfavgw2v['q2_tfidfavgw2v_features'].values.tolist(), index=processed_data_tfidfavgw2v.index)
            
            q1_features_headers = []
            q2_features_headers = []
            for i in range(300):
                q1_features_headers.append("q1_feat_"+str(i+1))
                q2_features_headers.append("q2_feat_"+str(i+1))
            
            q1_tfidfavgw2v_features.columns = q1_features_headers
            q2_tfidfavgw2v_features.columns = q2_features_headers
            
            logging.info("Changed column names for TFIDF Avg W2V features")
                                   
            processed_final_features = pd.concat([raw_data_featurized, processed_data_featurized.iloc[:,6:], q1_tfidfavgw2v_features, q2_tfidfavgw2v_features], axis=1)
            logging.info("Concatenated all DFs")
            
            processed_final_features = processed_final_features.iloc[:,6:]
            logging.info("Dropped unecessary columns")
            
            
            numerical_features = list(processed_final_features.columns)
            
            logging.info("Preprocessing is completed")
            

            save_object(
                file_path = self.data_transformation_config.preprocessed_data_file,
                obj = processed_final_features
            )
                        
            logging.info("Preprocessed data is saved")
            
            return (
                processed_final_features,
                numerical_features
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    
            
    def get_vectorizer_object(self, numerical_features):
        '''
        This function is responsible to return column vectorizer object
        '''
        try:
            numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"), ("normalizer", MinMaxScaler()))])

            vectorizer = ColumnTransformer([("num_pipeline", numerical_pipeline, numerical_features)])

            logging.info("vectorizer created")
                        
            return vectorizer
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, raw_data_path):
        try:
            logging.info('Obtaining preprocessed data')
            processed_final_features, numerical_features = self.get_preprocessed_data(raw_data_path) 
            logging.info('Obtained processed data')
            
            logging.info('Obtaining vectorizer object')
            vectorizer_obj = self.get_vectorizer_object(numerical_features)
            logging.info('Obtained vectorizer object')


            logging.info("Splitting data into train & test")
            
            X = processed_final_features.drop('is_duplicate', axis=1)
            y = processed_final_features['is_duplicate']
            logging.info("Split data in X & y")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)
            logging.info("Split data in train & test")
            
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            logging.info("Applying vectorizer object on processed data")
            X_train = vectorizer_obj.fit_transform(preprocessed_data)
            X_test = vectorizer_obj.transform(preprocessed_data)
                      
            logging.info("Vectorization is completed")
            
            save_object(
                file_path = self.data_transformation_config.vectorizer_obj_file,
                obj = vectorizer_obj
            )
            
            logging.info("Saved vectorizer object")
                        
            return (
                X_train,
                X_test,
                y_train,
                y_test,
                self.data_transformation_config.vectorizer_obj_file
            )
                        
        except Exception as e:
            raise CustomException(e, sys)