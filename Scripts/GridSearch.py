import numpy as np
"""
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
ros = RandomOverSampler(random_state=777)
smote = SMOTE(random_state=777)
"""
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
from sklearn.grid_search import GridSearchCV

pipe = Pipeline([('coltrans', ct), ('classifier', LogisticRegression())])

classifiers = {'logistic regression': {'classifier':[LogisticRegression()], 'classifier__penalty' : ['l1', 'l2'], 'classifier_C' : np.logspace(-4, 4, 20)},\
               'MultinomialNB': {'classifier': [MultinomialNB()], 'classifier__alpha': [0.01, 0.1, 1.0]},\
               'KNeigboursClassifier': {'classifier': [KNeighborsClassifier()], 'classifier__n_neighbors': [3, 7, 10], 'classifier__weights': ['uniform', 'distance']},\
               'SVC': {'classifier': [SVC()], }
               