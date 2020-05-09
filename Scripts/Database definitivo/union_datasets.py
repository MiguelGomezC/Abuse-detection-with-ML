#Fichero para unir los conjuntos de tweets:
#En el primero todos pertenecen a un periodo del día concreto (mañana),
#en el otro a hora viene dada mediante los segundos después de medianoche

#1:Madrugada- De 00:00 a 06:00
#2:Mañana- De 06:00 a 12:00
#3:Tarde- De 12:00 a 19:00
#4:Noche- De 19:00 a 00:00

import json

final_dataset = []

#Primer conjunto, separado en positivos y negativos

with open('negativos1','r') as fichero_negativos:
    for line in fichero_negativos:
        final_dataset.append([json.loads(line),2,0])
    
with open('positivos1','r') as fichero_positivos:
    for line in fichero_positivos:
        final_dataset.append([json.loads(line),2,1])

#Segundo conjunto

with open('database','r') as fichero2:
    data2 = json.load(fichero2)
    for line in data2:
        t = line[1]
        if t < 21600: #6*3600
            final_dataset.append([line[0],1,line[2]])
        elif t < 43200: #12*3600
            final_dataset.append([line[0],2,line[2]])
        elif t < 68400: #19*3600
            final_dataset.append([line[0],3,line[2]])
        else:
            final_dataset.append([line[0],4,line[2]])

with open('abusedetection_dataset','w') as ficherosalida:
    json.dump(final_dataset, ficherosalida)