from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

class TextTimeVectorizer():
    def fit_transform(self,X):
        tvec = TfidfVectorizer(ngram_range=(1,3), max_features=100000)
        text_features = tvec.fit_transform(X['text'])
        return hstack((text_features, X.time_class))