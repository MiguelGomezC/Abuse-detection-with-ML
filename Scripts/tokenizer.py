import json
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

with open('clean_data','r') as fichero:
    data = json.load(fichero)
df = pd.DataFrame(data, columns = ["text","time_class", "sentiment"])
X = df[['text','time_class']]
Y = df['sentiment']

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

def classification(pipeline, n_splits=5, X=X, Y=Y, average_method = 'macro'):
    """
    X: 2 column dataframe containing the text and time classes
    Y: dataframe column containing target sentiment values
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 777)
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
        print('              negative     positive')
        print('precision:',precision_score(Y[test], prediction, average=None))
        recall.append(recall_score(Y[test], prediction, average=average_method)*100)
        print('recall:   ',recall_score(Y[test], prediction, average=None))
        f1.append(f1_score(Y[test], prediction, average=average_method)*100)
        print('f1 score: ',f1_score(Y[test], prediction, average=None))
        print('-'*50)
    
    frame = pd.DataFrame({'y_Actual':Y[test],'y_Predicted':prediction}, columns = ['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(frame['y_Actual'], frame['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], normalize=True)
    
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
tvec = TfidfVectorizer(ngram_range=(1,3), max_features=100000) #Maybe tweak with smooth_idf?
column_trans = ColumnTransformer([('tfidf_vectorizer', TfidfVectorizer(), 'text')], remainder='passthrough') #Applies Tfidf to the text column and ignores the time class

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

#ROS_pipeline = make_pipeline(tvec, RandomOverSampler(random_state=777),lr)
#SMOTE_pipeline = make_pipeline(tvec, SMOTE(random_state=777),lr)

ROS_pipeline = make_pipeline(column_trans, RandomOverSampler(random_state=777),lr)
SMOTE_pipeline = make_pipeline(column_trans, SMOTE(random_state=777),lr)
