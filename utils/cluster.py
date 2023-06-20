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


def createClustersData(params='FDA',FDAwindow=1,root='data/idda/'):
    if FDAwindow%2==0:
        raise ValueError("You must use odd numbers as windows for FDA.")
    f=open(root+'clients_styles.json')
    clients=json.load(f)
    f.close()
    training_clients=[]
    test_clients=[]
    for key in clients:
        if key.find('U')>=0:
            training_clients.append(key)
        else:
            test_clients.append(key)

    if params=='mean':
        training_data=np.zeros((len(training_clients),3))
        test_data=np.zeros((len(test_clients),3))
        for i in range(len(training_clients)):
            training_data[i,:]=clients[training_clients[i]]['mean']
        for i in range(len(test_clients)):
            test_data[i,:]=clients[test_clients[i]]['mean']
    elif params=='mean_variance':
        training_data=np.zeros((len(training_clients),6))
        test_data=np.zeros((len(test_clients),6))
        for i in range(len(training_clients)):
            training_data[i,:3]=clients[training_clients[i]]['mean']
            training_data[i,3:]=clients[training_clients[i]]['variance']
        for i in range(len(test_clients)):
            test_data[i,:3]=clients[test_clients[i]]['mean']
            test_data[i,3:]=clients[test_clients[i]]['variance']
    elif params=='FDA':
        z=int(FDAwindow/2)
        training_data=np.zeros((len(training_clients),FDAwindow*(z+1)*3))
        test_data=np.zeros((len(test_clients),FDAwindow*FDAwindow*(z+1)*3))
        for i in range(len(training_clients)):
            training_data[i,:(z+1)*(z+1)*3]=np.array(clients[training_clients[i]]['FDA']['pos'])[:,:z+1,:z+1].reshape(1,-1)
            if z>0:
                training_data[i,(z+1)*(z+1)*3:]=np.array(clients[training_clients[i]]['FDA']['pos'])[:,-z:,:z+1].reshape(1,-1)
        
        for i in range(len(test_clients)):
            test_data[i,:(z+1)*(z+1)*3]=np.array(clients[test_clients[i]]['FDA']['pos'])[:,:z+1,:z+1].reshape(1,-1)
            if z>0:
                test_data[i,(z+1)*(z+1)*3:]=np.array(clients[test_clients[i]]['FDA']['pos'])[:,-z:,:z+1].reshape(1,-1)
        
    else:
        raise NotImplementedError
    return training_clients, training_data,test_clients,test_data


def createClusters(k,params='FDA',FDAwindow=1,root='data/idda/'):
    training_clients,training_data,test_clients,test_data=createClustersData(params,FDAwindow,root)
    #training|test_clients: name of clients
    #training|test_data: matrix (n_clients,size(data_per_client)) on each row, data for the corresponding client
    # client i in training_client correspond to training_data(i,:) data
    #same for test

    #MUST NORMALIZE!!!

    #cluster (fit on train, assign both to test and train)


    #return dictionary client_name, cluster number