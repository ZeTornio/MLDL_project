import json
import numpy as np
from PIL import Image

def extractMeansVarDictionary(root='data/idda/'):
    styles={}
    for fileName in ['test_diff_dom.json','test_same_dom.json','train.json']:
        print(fileName)
        f=open(root+fileName)
        clients=json.load(f)
        for key in clients:
            styles[key]={}
            a=None
            for im in clients[key]:
                img=Image.open(root+'images/'+im+'.jpg')
                if a is None:
                    a=np.asarray(img)
                    a=a[:,:,np.newaxis,:]/255
                else:
                    c=np.asarray(img)
                    c=c[:,:,np.newaxis,:]/255
                    a=np.concatenate((a,c),2)
            print(key)
            styles[key]['mean']=np.mean(a,(0,1,2)).tolist()
            styles[key]['variance']=np.var(a,(0,1,2)).tolist()
        f.close()
    return styles

from utils.FDA import extractClientsStyles
def extractFDAstylesDict(n,root='data/idda/',size=(540,960)):
    stylesFDA={}
    for fileName in ['test_diff_dom.json','test_same_dom.json','train.json']:
        stylesFDA.update(extractClientsStyles(n,root+fileName,root+'images/',size))
    for key in stylesFDA:
        fda={}
        fda['pos']=stylesFDA[key]['pos'].numpy().tolist()
        fda['neg']=stylesFDA[key]['neg'].numpy().tolist()
        stylesFDA[key]={}
        stylesFDA[key]['FDA']=fda
    return stylesFDA