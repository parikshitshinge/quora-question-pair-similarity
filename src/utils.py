import os 
import sys
from src.exception import CustomException
from src.logger import logging
import dill

import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from thefuzz import fuzz

# Preprocessing
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('punkt')
from thefuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
# Metrics
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'wb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = log_loss(y_true, y_pred)(y_train, y_train_pred)
            test_model_score = log_loss(y_true, y_pred)(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)


# Preprocess
def preprocess(x):
    STOPWORDS = set(stopwords.words('english'))
    STOPWORDS.remove('not')

    # 0. Convert to string
    x = str(x)
    
    # 1. Lower casing
    x = x.lower()
    
    # 2. Remove HTML tags
    re_html_tag = re.compile(r'<.*?>')
    x = re_html_tag.sub(r'', x)
    
    # 3. Remove URLs
    re_url = re.compile(r'https://\S+|www\.\S+')
    x = re_url.sub(r'', x)
    
    # 4. Expand contractions
    x = x.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                        .replace("€", " euro ").replace("'ll", " will")    
    
    # 5. Number to units
    x = x.replace(",000,000,000", "b").replace(",000,000", "m").replace(",000", "k")
    
    # 6. Remove punctuations
    x = "".join([i for i in x if i not in string.punctuation ])
    
    # 7. Remove stopwords
    x = " ".join([word for word in x.split() if word not in STOPWORDS])
    
    # 8. Lemmatization
    lemmatizer = WordNetLemmatizer()
    x = " ".join([lemmatizer.lemmatize(word) for word in x.split()])
	
    return x


# Feature engineering functions
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.remove('not')

def compute_common_tokens_count(row):
    """
    Returns number of common tokens in Question1 & Question2
    
    """
    t1 = set(map(lambda word: str(word).lower().strip(), word_tokenize(row['question1']) ))
    t2 = set(map(lambda word: str(word).lower().strip(), word_tokenize(row['question2']) ))  
    return len(t1 & t2)

def compute_common_tokens_share(row):
    """
    Returns percentage of common tokens in Question1 & Question2 i.e. common tokens / total tokens
    
    """
    t1 = set(map(lambda word: str(word).lower().strip(), word_tokenize(row['question1']) ))
    t2 = set(map(lambda word: str(word).lower().strip(), word_tokenize(row['question2']) ))  
    return len(t1 & t2) / (len(t1)+len(t2))

def compute_common_words_count(row):
    """
    Returns number of common words in Question1 & Question2
    
    """
    w1 = set(map(lambda word: str(word).lower().strip(), row['question1'].split(" ") ))
    w2 = set(map(lambda word: str(word).lower().strip(), row['question2'].split(" ") ))    
    return len(w1 & w2)   

def compute_common_words_share(row):
    """
    Returns percentage of common words in Question1 & Question2 i.e. common words / total words
    
    """
    w1 = set(map(lambda word: str(word).lower().strip(), row['question1'].split(" ") ))
    w2 = set(map(lambda word: str(word).lower().strip(), row['question2'].split(" ") ))    
    return len(w1 & w2) / (len(w1) + len(w2))

def compute_common_nonstopwords_count(row):
    """
    Returns number of common nonstopwords in Question1 & Question2
    
    """
    w1 = set(map(lambda word: str(word).lower().strip(), [word for word in row['question1'].split(" ") if word not in STOPWORDS] ))
    w2 = set(map(lambda word: str(word).lower().strip(), [word for word in row['question2'].split(" ") if word not in STOPWORDS] ))    
    return len(w1 & w2)    

def compute_common_nonstopwords_share(row):
    """
    Returns percentage of common nonstopwords in Question1 & Question2 i.e. common nonstopwords / total nonstopwords
    
    """
    w1 = set(map(lambda word: str(word).lower().strip(), [word for word in row['question1'].split(" ") if word not in STOPWORDS] ))
    w2 = set(map(lambda word: str(word).lower().strip(), [word for word in row['question2'].split(" ") if word not in STOPWORDS] ))    
    return len(w1 & w2) / (len(w1) + len(w2))


# Feature engineering basic 
def feature_engg_basic(df):
    """
    Returns set of features:
	q1_frequency = Frequency of Question1 in train corpus 
	q2_frequency = Frequency of Question2 in train corpus 
	q1_len = Length of Question1 
	q2_len = Length of Question2 
	q1_tokens_count = # of Tokens in Question1 
	q2_tokens_count = # of Tokens in Question2 
	q1_words_count = # of Words in Question1 
	q2_words_count = # of Words in Question2 
	q1_nonstopwords_count = # of Non-Stopwords in Question1 
	q2_nonstopwords_count = # of Non-Stopwords in Question2 
	common_tokens_count = # of Common Tokens in Question1 & Question2 
	common_tokens_share = (# of Common Tokens in Question1 & Question2) / (Total Tokens in Question1 & Question2) 
	common_words_count = # of Common Words in Question1 & Question2 
	common_words_share = (# of Common Words in Question1 & Question2) / (Total Words in Question1 & Question2) 
	common_nonstopwords_count = # of Common Non-Stopwords in Question1 & Question2 
	common_nonstopwords_share = (# of Common Non-Stopwords in Question1 & Question2) / (Total Non-Stopwords in Question1 & Question2)     
    
    """
    # Convert all questions to string first
    df['question1'] = df['question1'].apply(lambda x: str(x))
    df['question2'] = df['question2'].apply(lambda x: str(x))
    
    # q1_frequency
    start = datetime.now()
    df['q1_frequency'] = df.groupby(by='qid1')['qid1'].transform('count')
    print("Feature 'q1_frequency' created. Time taken: {0}".format(datetime.now() - start))
    
    # q2_frequency
    start = datetime.now()
    df['q2_frequency'] = df.groupby(by='qid2')['qid2'].transform('count')
    print("Feature 'q2_frequency' created. Time taken: {0}".format(datetime.now() - start))
    
    # q1_length
    start = datetime.now()
    df['q1_length'] = df['question1'].str.len()
    print("Feature 'q1_length' created. Time taken: {0}".format(datetime.now() - start))
    
    # q2_length
    start = datetime.now()
    df['q2_length'] = df['question2'].str.len()
    print("Feature 'q2_length' created. Time taken: {0}".format(datetime.now() - start))
    
    # q1_tokens_count
    start = datetime.now()
    df['q1_tokens_count'] = df['question1'].apply(lambda x: len(word_tokenize(x)))
    print("Feature 'q1_tokens_count' created. Time taken: {0}".format(datetime.now() - start))
    
    # q2_tokens_count
    start = datetime.now()
    df['q2_tokens_count'] = df['question2'].apply(lambda x: len(word_tokenize(x)))
    print("Feature 'q2_tokens_count' created. Time taken: {0}".format(datetime.now() - start))

    # q1_words_count
    start = datetime.now()
    df['q1_words_count'] = df['question1'].apply(lambda x: len(x.split(" ")))
    print("Feature 'q1_words_count' created. Time taken: {0}".format(datetime.now() - start))

    # q2_words_count
    start = datetime.now()
    df['q2_words_count'] = df['question2'].apply(lambda x: len(x.split(" ")))
    print("Feature 'q2_words_count' created. Time taken: {0}".format(datetime.now() - start))
    
    # q1_nonstopwords_count
    start = datetime.now()
    df['q1_nonstopwords_count'] = df['question1'].apply(lambda x: len([word for word in str.lower(x).split(" ") if word not in STOPWORDS]))
    print("Feature 'q1_nonstopwords_count' created. Time taken: {0}".format(datetime.now() - start))
    
    # q2_nonstopwords_count
    start = datetime.now()
    df['q2_nonstopwords_count'] = df['question2'].apply(lambda x: len([word for word in str.lower(x).split(" ") if word not in STOPWORDS]))   
    print("Feature 'q2_nonstopwords_count' created. Time taken: {0}".format(datetime.now() - start)) 
    
    # common_tokens_count
    start = datetime.now()
    df['common_tokens_count'] =  df.apply(compute_common_tokens_count, axis=1)
    print("Feature 'common_tokens_count' created. Time taken: {0}".format(datetime.now() - start))
    
    # common_tokens_share
    start = datetime.now()
    df['common_tokens_share'] =  df.apply(compute_common_tokens_share, axis=1)
    print("Feature 'common_tokens_share' created. Time taken: {0}".format(datetime.now() - start))
    
    # common_words_count
    start = datetime.now()
    df['common_words_count'] =  df.apply(compute_common_words_count, axis=1)
    print("Feature 'common_words_count' created. Time taken: {0}".format(datetime.now() - start))
        
    # common_words_share
    start = datetime.now()
    df['common_words_share'] =  df.apply(compute_common_words_share, axis=1)
    print("Feature 'common_words_share' created. Time taken: {0}".format(datetime.now() - start))
    
    # common_nonstopwords_count
    start = datetime.now()
    df['common_nonstopwords_count'] =  df.apply(compute_common_nonstopwords_count, axis=1)
    print("Feature 'common_nonstopwords_count' created. Time taken: {0}".format(datetime.now() - start))
    
    # common_nonstopwords_share
    start = datetime.now()
    df['common_nonstopwords_share'] =  df.apply(compute_common_nonstopwords_share, axis=1)
    print("Feature 'common_nonstopwords_share' created. Time taken: {0}".format(datetime.now() - start))
    
    
# Advanced feature engineering
SAFE_DIVISION = 0.0001

def compute_common_tokens_count_min(row):
    """
    Returns number of common tokens in Question1 & Question2
    
    """
    t1 = set(map(lambda word: str(word).lower().strip(), word_tokenize(row['question1']) ))
    t2 = set(map(lambda word: str(word).lower().strip(), word_tokenize(row['question2']) ))  
    return len(t1 & t2) / (min(len(t1), len(t2)) + SAFE_DIVISION)

def compute_common_tokens_count_max(row):
    """
    Returns number of common tokens in Question1 & Question2
    
    """
    t1 = set(map(lambda word: str(word).lower().strip(), word_tokenize(row['question1']) ))
    t2 = set(map(lambda word: str(word).lower().strip(), word_tokenize(row['question2']) ))  
    return len(t1 & t2) / (max(len(t1), len(t2)) + SAFE_DIVISION)


def compute_common_words_count_min(row):
    """
    Returns number of common words in Question1 & Question2
    
    """
    w1 = set(map(lambda word: str(word).lower().strip(), row['question1'].split(" ") ))
    w2 = set(map(lambda word: str(word).lower().strip(), row['question2'].split(" ") ))    
    return len(w1 & w2) / (min(len(w1), len(w2)) + SAFE_DIVISION)

def compute_common_words_count_max(row):
    """
    Returns number of common words in Question1 & Question2
    
    """
    w1 = set(map(lambda word: str(word).lower().strip(), row['question1'].split(" ") ))
    w2 = set(map(lambda word: str(word).lower().strip(), row['question2'].split(" ") ))    
    return len(w1 & w2) / (max(len(w1), len(w2)) + SAFE_DIVISION)

def compute_common_nonstopwords_count_min(row):
    """
    Returns number of common nonstopwords in Question1 & Question2
    
    """
    w1 = set(map(lambda word: str(word).lower().strip(), [word for word in row['question1'].split(" ") if word not in STOPWORDS] ))
    w2 = set(map(lambda word: str(word).lower().strip(), [word for word in row['question2'].split(" ") if word not in STOPWORDS] ))    
    return len(w1 & w2) / (min(len(w1), len(w2)) + SAFE_DIVISION)

def compute_common_nonstopwords_count_max(row):
    """
    Returns number of common nonstopwords in Question1 & Question2
    
    """
    w1 = set(map(lambda word: str(word).lower().strip(), [word for word in row['question1'].split(" ") if word not in STOPWORDS] ))
    w2 = set(map(lambda word: str(word).lower().strip(), [word for word in row['question2'].split(" ") if word not in STOPWORDS] ))    
    return len(w1 & w2) / (max(len(w1), len(w2)) + SAFE_DIVISION)



    # fuzz_partial_ratio
    start = datetime.now()
    df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(x['question1'], x['question2']), axis=1 )
    print("Feature 'fuzz_partial_ratio' created. Time taken: {0}".format(datetime.now() - start))
    
    # fuzz_token_sort_ratio
    start = datetime.now()
    df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(x['question1'], x['question2']), axis=1 )
    print("Feature 'fuzz_token_sort_ratio' created. Time taken: {0}".format(datetime.now() - start))
    
    # fuzz_token_set_ratio
    start = datetime.now()
    df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(x['question1'], x['question2']), axis=1 )
    print("Feature 'fuzz_token_set_ratio' created. Time taken: {0}".format(datetime.now() - start))

    # fuzz_partial_token_sort_ratio
    start = datetime.now()
    df['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(x['question1'], x['question2']), axis=1 )
    print("Feature 'fuzz_partial_token_sort_ratio' created. Time taken: {0}".format(datetime.now() - start))
    
    # common_tokens_count_min
    start = datetime.now()
    df['common_tokens_count_min'] =  df.apply(compute_common_tokens_count_min, axis=1)
    print("Feature 'common_tokens_count_min' created. Time taken: {0}".format(datetime.now() - start))
    
    # common_tokens_count_max
    start = datetime.now()
    df['common_tokens_count_max'] =  df.apply(compute_common_tokens_count_max, axis=1)
    print("Feature 'common_tokens_count_max' created. Time taken: {0}".format(datetime.now() - start))
    
    # common_words_count_min
    start = datetime.now()
    df['common_words_count_min'] =  df.apply(compute_common_words_count_min, axis=1)
    print("Feature 'common_words_count_min' created. Time taken: {0}".format(datetime.now() - start))
        
    # common_words_count_max
    start = datetime.now()
    df['common_words_count_max'] =  df.apply(compute_common_words_count_max, axis=1)
    print("Feature 'common_words_count_max' created. Time taken: {0}".format(datetime.now() - start))
    
    # common_nonstopwords_count_min
    start = datetime.now()
    df['common_nonstopwords_count_min'] =  df.apply(compute_common_nonstopwords_count_min, axis=1)
    print("Feature 'common_nonstopwords_count_min' created. Time taken: {0}".format(datetime.now() - start))
    
    # common_nonstopwords_count_max
    start = datetime.now()
    df['common_nonstopwords_count_max'] =  df.apply(compute_common_nonstopwords_count_max, axis=1)
    print("Feature 'common_nonstopwords_count_max' created. Time taken: {0}".format(datetime.now() - start))
    
# TFIDF Avg W2V
def compute_tfidf_avg_w2v(question):
    question = str(question)
        
    question_nlp = nlp(question)
    avg_vector = np.zeros([len(question_nlp), len(question_nlp[0].vector)]) # Initialize the mean vector with 0s
    
    for word in question_nlp:
        word_vector = word.vector
        try:
            idf = word2tfidf[str(word)]
        except:
            idf = 0
        avg_vector += word_vector * idf
    return avg_vector.mean(axis=0)