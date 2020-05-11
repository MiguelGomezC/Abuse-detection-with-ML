from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def TexTimeVectorizer(X):
    tvec = TfidfVectorizer(ngram_range=(1,3), max_features=100000)
    fit_text = tvec.fit_transform(X['text'])
    return 