import pandas as pd
from pandas.plotting import table
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import json

cereal_df = pd.read_csv("/tmp/tmp07wuam09/data/cereal.csv")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def classification(pipeline, title, n_splits=5, X=X, Y=Y, average_method = 'macro'):
    """
    X: 2 column dataframe containing text and time features
    Y: dataframe column containing target sentiment values
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 777)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    predictions = np.array([])
    y_values = np.array([])
    for train, test in kfold.split(X,Y):
        X_train, X_test = X.iloc[train], X.iloc[test]
        Y_train, Y_test = Y.iloc[train], Y.iloc[test]
        y_values = np.concatenate((y_values,Y_test))
        model_fit = pipeline.fit(X_train, Y_train)
        prediction = model_fit.predict(X_test)
        predictions = np.concatenate((predictions,prediction))
        scores = model_fit.score(X_test,Y_test)
        accuracy.append(scores * 100)
        precision.append(precision_score(Y_test, prediction, average=average_method)*100)
        print('              negative     positive')
        print('precision:',precision_score(Y_test, prediction, average=None))
        recall.append(recall_score(Y_test, prediction, average=average_method)*100)
        print('recall:   ',recall_score(Y_test, prediction, average=None))
        f1.append(f1_score(Y_test, prediction, average=average_method)*100)
        print('f1 score: ',f1_score(Y_test, prediction, average=None))
        print('-'*50)

    frame = pd.DataFrame({'y_Actual':y_values,'y_Predicted':predictions}, columns = ['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(frame['y_Actual'], frame['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], normalize='index')
    sn.heatmap(confusion_matrix, annot=True)
    plt.title(title)
    fig = plt.gcf()
    images_dir = 'Imágenes para la memoria'
    fig.savefig(f"{images_dir}/{title}.png")
    plt.show()
    
    report = classification_report(y_values,predictions,output_dict=True)
    dataframe = pd.DataFrame(report).transpose()
    results_dir ='Reportes de clasificación'
    with open(f"{results_dir}/{title}.tex", 'w') as fichero:
        fichero.write(title+'\n')
        fichero.write(dataframe.to_latex())
    
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))

#Vectorización de la característica tiempo via MinMaxScaler y texto via TFIDF
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Scaler",MinMaxScaler(), ['time_feature']),\
                        ("Tfidfvectorizer", TfidfVectorizer(ngram_range=(1,3), max_features=100000),'text')])

#Función para crear una pipeline a partir de un método (obj)
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

def SMOTE_pipeline(obj):
    return make_pipeline(ct, SMOTE(random_state=777), obj)

#Importar ahora los métodos que vamos a utilizar
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

if __name__ == "__main__":
    #Cargar la lista de métodos con los parámetros ajustados a partir de GridSearch.py
    import joblib
    methods = joblib.load('Resultados Gridsearch\imb_params.pkl')
    for method, score in methods:
        title = str(method['classifier']).split('(')[0]
        #input('Presione enter para aplicar el siguiente método')
        print('Se está evaluando el método', title,'\nObtuvo una puntuación de ',score,' en la búsqueda de rejilla.\n')
        classification(SMOTE_pipeline(method['classifier']) ,title)
        print('\n')