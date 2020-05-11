import json
import pandas as pd
import numpy as np
from TexTimeVectorizer import TexTimeVectorizer as ttv

with open('clean_data','r') as fichero:
    data = json.load(fichero)
df = pd.DataFrame(data, columns = ["text","time class", "sentiment"])

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

def classification(n_splits, X, Y, vectorizer, pipeline, average_method):
    """
    X: 2 column array containing the text and time classes
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 777) #clean_data shuffled previously
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train, test in kfold.split(X,Y):
        model_fit = pipeline.fit(X[train], Y[train])
        prediction = model_fit.predict(X[test])
        scores = model_fit.score(X[test],Y[test])
        accuracy.append(scores * 100)
        precision.append(precision_score(Y[test], prediction, average=average_method)*100)
        print('              negative    neutral     positive')
        print('precision:',precision_score(Y[test], prediction, average=None))
        recall.append(recall_score(Y[test], prediction, average=average_method)*100)
        print('recall:   ',recall_score(Y[test], prediction, average=None))
        f1.append(f1_score(Y[test], prediction, average=average_method)*100)
        print('f1 score: ',f1_score(Y[test], prediction, average=None))
        print('-'*50)

    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
tvec = TfidfVectorizer(ngram_range=(1,3), max_features=100000) #Maybe tweak with smooth_idf?

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

ROS_pipeline = make_pipeline(tvec, RandomOverSampler(random_state=777),lr)
SMOTE_pipeline = make_pipeline(tvec, SMOTE(random_state=777),lr)