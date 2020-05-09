import spacy
import json
import re
import random

with open('abusedetection_dataset') as fichero:
    datos = json.load(fichero)

nlp = spacy.load('es_core_news_sm')

def datacleaner(text):
    text = text.lower()
    parsed = nlp(text)
    final_tokens = []
    for t in parsed:
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t)[0] == '@':
            pass
        else:
            sc_removed = re.sub("[^a-zñáéíóúäëïöü]", '', str(t.lemma_))
            if len(sc_removed) > 1:
                final_tokens.append(sc_removed)
    joined = ' '.join(final_tokens)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    return spell_corrected

datos_clean = [[datacleaner(line[0]),line[1],line[2]] for line in datos]
random.shuffle(datos_clean)

with open('clean_data','w') as ficherosalida:
    json.dump(datos_clean, ficherosalida)