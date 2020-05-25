#Archivo para crear un nuevo corpus de respuestas de varias horas del día

import json
from random import sample
homepath = "D:\TFG\Descargadordetweets\\"
Archivos = ["Diadelamujer_1_6_Marzo_3.2020-03-08_04","Diadelamujer_1_6_Marzo_3.2020-03-08_05","Diadelamujer_1_6_Marzo_3.2020-03-08_06","Diadelamujer_1_6_Marzo_3.2020-03-08_07","Diadelamujer_1_6_Marzo_3.2020-03-08_08","Diadelamujer_1_6_Marzo_3.2020-03-08_09","Diadelamujer_1_6_Marzo_3.2020-03-08_10",\
            "Diadelamujer_1_6_Marzo_3.2020-03-08_11","Diadelamujer_1_6_Marzo_3.2020-03-08_12","Diadelamujer_1_6_Marzo_3.2020-03-08_13","Diadelamujer_1_6_Marzo_3.2020-03-08_14","Diadelamujer_1_6_Marzo_3.2020-03-08_15","Diadelamujer_1_6_Marzo_3.2020-03-08_16","Diadelamujer_1_6_Marzo_3.2020-03-08_17",\
            "Diadelamujer_1_6_Marzo_3.2020-03-08_18","Diadelamujer_1_6_Marzo_3.2020-03-08_19","Diadelamujer_1_6_Marzo_3.2020-03-08_20","Diadelamujer_1_6_Marzo_3.2020-03-08_21","Diadelamujer_1_6_Marzo_3.2020-03-08_22","Diadelamujer_1_6_Marzo_3.2020-03-08_23","Diadelamujer_1_6_Marzo_3.2020-03-09_00",\
            "Diadelamujer_1_6_Marzo_3.2020-03-09_01","Diadelamujer_1_6_Marzo_3.2020-03-09_02"]

if __name__ == '__main__':
    FinalSet = []
    for index in range(23):
        Datos = []
        with open(homepath+Archivos[index]) as fichero:
            print('Empezando la selección en el archivo', Archivos[index])
            for line in fichero:
                file = json.loads(line)
                if (file['lang']=='es' or file['lang']=='und') and file['in_reply_to_user_id'] != None:
                    Datos.append(file)
            rand = sample(Datos,87)
            FinalSet.extend(rand)
            print('Sección',index,'añadida al conjunto final de datos')
  
with open('D:\TFG\Onlyreplies\repliesrawdict','w',encoding = 'utf-8') as archivosalida:
    json.dump(FinalSet, archivosalida) #Se codifica UNA lista de diccionarios
