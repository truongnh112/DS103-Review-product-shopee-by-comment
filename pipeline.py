import re
import json
import requests
import pandas as pd
import glob
import preprocessing
from preprocessing import *
from pyvi import ViTokenizer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline


module_count_vector = CountVectorizer()
model_rf_preprocess = Pipeline([('vect', module_count_vector),
                    ('tfidf', TfidfTransformer()),
                    ])

f = open('/Users/nguyen/Documents/DemoDs103/preprocess_cmt.txt', 'r')
preprocess_data = f.read().split('\n')

print(type(preprocess_data))
#print(preprocess_data)


model_rf_preprocess.fit_transform(preprocess_data)
print(f"Số từ trong từ điển: {len(module_count_vector.vocabulary_)}")

def encode_cmt(raw_cmt):
    raw_cmt = pd.Series([raw_cmt])
    pre_cmt = list(map(text_preprocess, raw_cmt))
    fin_cmt = model_rf_preprocess.transform(pre_cmt)
    return fin_cmt

def encode_list(list_cmt):
    pre_cmt = list(map(text_preprocess, list_cmt))
    fin_cmt = model_rf_preprocess.transform(pre_cmt)
    return fin_cmt
 


def predict_raw(model, raw_cmt):
    # tiền xử lý dữ liệu sử dụng module model_rf_preprocess. 
    fin_cmt = encode_cmt(raw_cmt=raw_cmt)
    # phán đoán nhãn
    pred = model.predict(fin_cmt)
    if pred[0] == -2:
        return "-2. Bình luận rất tiêu cực!"
    elif pred[0] == -1:
        return "-1. Bình luận hơi tiêu cực!"
    elif pred[0] == 0:
        return "0. Bình luận trung tính (bình thường)."
    elif pred[0] == 1:
        return "1. Bình luận hơi tích cực!"
    elif pred[0] == 2:
        return "2. Bình luận rất tích cực!"