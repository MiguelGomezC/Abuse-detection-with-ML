import json
from os import listdir

def secs_after_midnight(string):
    hours, minutes, seconds = tuple(map(int, string.split(':')))
    return hours*3600+minutes*60+seconds

def categorizer(datos, initial=0):
    c=initial
    for line in datos[c:]:
        print('Expression number',c,'\n'+line[0]+'\n')
        try:
            a = input('Introduce a value:')
            while not(a=='0' or a=='1' or a=='Stop'):
                a = input('Introduce a value:')
            if a=='Stop':
                print('You reached the value', c)
                return c
            line.append(int(a))
            c+=1
        except:
            print('There has been an exception')
            pass
    return c

if __name__=="__main__":
    
    datos=[]
    
    with open('repliesrawdict','r') as fichero:
        rawlist = json.load(fichero)
    for line in rawlist:
        if 'extended_tweet' in line:
            a = [line['extended_tweet']['full_text'], secs_after_midnight((line['created_at'].split())[3])]
        else:
            a = [line['text'], secs_after_midnight(line['created_at'].split()[3])]
        datos.append(a)
    b = 1999
    c = categorizer(datos,initial=b)
    files = listdir('C:\\Users\Miguel\Documents\GitHub\Abuse-detection-with-ML\Scripts\RepliesOnly')
    a = 0
    name = 'labeled_data0'
    while name in files:
        a+=1
        name = name[:12]+str(a)
    with open(name,'w',encoding='utf-8') as out:
        json.dump(datos[b:c],out)