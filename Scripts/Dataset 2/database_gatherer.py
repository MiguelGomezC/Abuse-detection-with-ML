#Script para juntar todas las semilistas en una sola

from os import listdir
import json

if __name__=="__main__":
    names = []
    for name in listdir('C:\\Users\Miguel\Documents\TFG\Onlyreplies'):
        if name[:12] == 'labeled_data':
            names.append(name)
    data = []
    for name in names:
        with open(name,'r') as fichero:
            data.extend(json.load(fichero))
    
    with open('database', 'w') as fichero:
        json.dump(data, fichero)