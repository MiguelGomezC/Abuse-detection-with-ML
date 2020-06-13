import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline as imbpipe
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=777)

#Carga de datos
import json
with open('clean_data','r') as fichero:
    data = json.load(fichero)
df = pd.DataFrame(data, columns = ["text","time_feature", "sentiment"])
X = df[['text','time_feature']]
Y = df['sentiment']

from sklearn.pipeline import Pipeline

#Vectorización de la característica tiempo via MinMaxScaler y texto via TFIDF
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Scaler", MinMaxScaler(), ['time_feature']),\
                        ("Tfidfvectorizer", TfidfVectorizer(ngram_range=(1,3), max_features=50000),'text')])

#Métodos que vamos a utilizar
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(max_features = "auto", max_depth = None)

def pipe(obj):
    obj = classifiers[obj]['classifier'][0]
    return Pipeline([('coltrans', ct), ('classifier', obj)])

def resampling_pipe(obj):
    obj = classifiers[obj]['classifier'][0]
    return imbpipe([('coltrans', ct),('SMOTE', SMOTE()), ('classifier', obj)])

#Diccionario con los parámetros a probar en la búsqueda por rejilla
classifiers = {'LogisticRegression': {'classifier':[LogisticRegression()], 'classifier__penalty' : ['l1', 'l2'], 'classifier__C' : np.logspace(-4, 4, 20)},\
               'MultinomialNB': {'classifier': [MultinomialNB()], 'classifier__alpha': [0.01, 0.1, 1.0]},\
               'KNeighborsClassifier': {'classifier': [KNeighborsClassifier()], 'classifier__n_neighbors': range(1,31), 'classifier__weights': ['uniform', 'distance']},\
               'SVC': {'classifier': [SVC()],'classifier__C':[1,10,100,1000],'classifier__gamma':[1,0.1,0.001,0.0001], 'classifier__kernel':['linear','rbf']},\
               'RandomForestClassifier':{'classifier':[RandomForestClassifier()],'classifier__bootstrap': [True],'classifier__max_depth': range(5,32,2), 'classifier__max_features': [2, 3],'classifier__min_samples_leaf': [3, 4, 5],'classifier__min_samples_split': [8, 10, 12],'classifier__n_estimators': [100, 200, 300, 1000]},\
               'GradientBoostingClassifier':{'classifier':[GradientBoostingClassifier()],'classifier__n_estimators':range(20,81,10),'classifier__max_depth':range(5,16,2), 'classifier__min_samples_split':range(200,1001,200)},\
               'AdaBoostClassifier':{'classifier':[AdaBoostClassifier()], 'classifier__base_estimator':[DTC], 'classifier__base_estimator__max_depth':range(5,32,2), 'classifier__n_estimators':range(20,81,10)}}
    
#SVC: recall: C=10, gamma = 1,  kernel='linear'; accuracy: C=100, gamma=1, kernel = 'linear'
#MultinomialNB: accuracy: alpha = 1; recall: alpha = 0.01
def grid_search(classifier):
    """
    --NO RESAMPLING--
    """
    param_grid = (classifiers[classifier])
    pipeline = pipe(classifier)
    evaluation = GridSearchCV(pipeline, param_grid, scoring = 'recall', refit=True, n_jobs=-1, verbose = 2)
    evaluation.fit(X,Y)
    print("La mejor combinación ha sido: ", evaluation.best_params_, "\nCon una puntuación de: ", evaluation.best_score_)
    return (evaluation.best_params_, evaluation.best_score_)

def imb_grid_search(classifier):
    """
    --WITH RESAMPLING--
    """
    param_grid = (classifiers[classifier])
    pipeline = resampling_pipe(classifier)
    evaluation = GridSearchCV(pipeline, param_grid, scoring = 'recall', refit=True, n_jobs=-1, verbose = 2)
    evaluation.fit(X,Y)
    print("La mejor combinación ha sido: ", evaluation.best_params_, "\nCon una puntuación de: ", evaluation.best_score_)
    return (evaluation.best_params_, evaluation.best_score_)

if __name__ == "__main__":
    import joblib
    """
    params = []
    for key in classifiers.keys():
        GS = grid_search(key)
        params.append(GS) #Pares parámetros, puntuación
    joblib.dump(params, 'Resultados Gridsearch\params.pkl')
        
    params = []
    for key in classifiers.keys():
        GS = imb_grid_search(key)
        params.append(GS)
    joblib.dump(params, 'Resultados Gridsearch\imb_params.pkl')
    """