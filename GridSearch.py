import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
"""
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
ros = RandomOverSampler(random_state=777)
smote = SMOTE(random_state=777)
"""
import json
with open('clean_data','r') as fichero:
    data = json.load(fichero)
df = pd.DataFrame(data, columns = ["text","time_feature", "sentiment"])
X = df[['text','time_feature']]
Y = df['sentiment']

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Scaler", MinMaxScaler(), ['time_feature']),\
                        ("Tfidfvectorizer", TfidfVectorizer(ngram_range=(1,3), max_features=50000),'text')])
transformerparams = {'coltrans':[ct],'coltrans__TfidfVectorizer__ngram_range':[(1,1),(1,2),(1,3)],\
               'coltrans__TfidfVectorizer__max_features':[50000]}
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
#probar gradient boost, adaptative boost

def pipe(obj):
    obj = classifiers[obj]['classifier'][0]
    return Pipeline([('coltrans', ct), ('classifier', obj)])

classifiers = {'LogisticRegression': {'classifier':[LogisticRegression()], 'classifier__penalty' : ['l1', 'l2'], 'classifier_C' : np.logspace(-4, 4, 20)},\
               'MultinomialNB': {'classifier': [MultinomialNB()], 'classifier__alpha': [0.01, 0.1, 1.0]},\
               'KNeighborsClassifier': {'classifier': [KNeighborsClassifier()], 'classifier__n_neighbors': [1,3,5,7,10,13,15,20,25,30], 'classifier__weights': ['uniform', 'distance']},\
               'SVC': {'classifier': [SVC()],'classifier__C':[1,10,100,1000],'classifier__gamma':[1,0.1,0.001,0.0001], 'classifier__kernel':['linear','rbf']},\
               'RandomForestClassifier':{'classifier':[RandomForestClassifier()],'classifier__bootstrap': [True],'classifier__max_depth': [80, 90, 100, 110], 'classifier__max_features': [2, 3],'classifier__min_samples_leaf': [3, 4, 5],'classifier__min_samples_split': [8, 10, 12],'classifier__n_estimators': [100, 200, 300, 1000]}}
#SVC: recall: C=10, gamma = 1,  kernel='linear'; accuracy: C=100, gamma=1, kernel = 'linear'
#MultinomialNB: accuracy: alpha = 1; recall: alpha = 0.01
def grid_search(classifier):
    """
    (Returns best estimator)
    --NO RESAMPLING--
    """
    param_grid = (classifiers[classifier])
    pipeline = pipe(classifier)
    evaluation = GridSearchCV(pipeline, param_grid, scoring = 'recall', refit=True, n_jobs=-1, verbose = 2)
    evaluation.fit(X,Y)
    return evaluation.best_estimator_

from main import classification
