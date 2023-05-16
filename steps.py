from datasets.idda import IDDADataset
from client import Client
from server import Server
import json

class Args:
    def __init__(self,num_rounds,num_epochs,clients_per_round=1,hnm=False,lr=0.05,bs=8,wd=0,m=0.9,change_lr_m=None,change_type='new_val',change_time='at_given_rounds'):
        #Rounds 
        self.num_rounds=num_rounds
        #Epochs per client for each round
        self.num_epochs=num_epochs
        #Clients selected per round
        self.clients_per_round=clients_per_round
        #Use HardMargin
        self.hnm=hnm
        #Batch size
        self.bs=bs
        #Learning rate
        self.lr=lr
        #Weight decay
        self.wd=wd
        #Momentum
        self.m=m
        #Change lr and m after tot: format could be 
        # if from_start: (int rounds, new params) OR (float rounds wrt total, new params) rounds
        # if from previous: (int rounds, new params) OR (float rounds wrt total, new params)
        self.change_lr_m=change_lr_m
        self.progressive_lr_m=self.getParams(change_type,change_time)

    def getParams(self,changeType,changeTime):

        '''
        changeTime:
        If at_given_rounds the round number is considered from the beginning.
        If after_given_rounds the round number is considered from the previous change.
        If cyclical the number is considered from the previous change.
            Beware that in that case the initial parameters are kept only the first round. If you want to cycle between two ore more parameters
            like 0-[10 rounds]->1-[20 rounds]->0-[10 rounds]->1->... you should put
            0 as initial parameter, then change_lr_m=[(10,1),(20,0)]
            If you use only one parameter in cyclical, you will perform the same drop each -given- rounds. (working but non-sense if new_val)
        changeType:
        If new_val, it will be set as the value.
        If abs_drop, the amount subctracted to the initial value. Beware that it could become negative.
        If fact_drop, the factor for which the value is divided.
        '''
        if self.change_lr_m==None:
            return [(0,self.num_rounds,self.lr,self.m)]
        params=[]
        start=0
        lr=self.lr
        m=self.m
        if changeType=='new_val':
            change=lambda old_val,change: change
        elif changeType=='abs_drop':
            change=lambda old_val,val:old_val-val
        elif changeType=='fact_drop':
            change=lambda old_val,change:old_val/change
        else:
            raise NotImplementedError
        j=0
        if changeTime=='at_given_rounds':
            for i in range(0,len(self.change_lr_m)):
                j=min(max(self.change_lr_m[i][0],j),self.num_rounds)
                params.append((start,j,lr,m))
                lr=change(lr,self.change_lr_m[i][1])
                m=change(m,self.change_lr_m[i][2]) if len(self.change_lr_m[i])==3 else m
                start=j
            params.append((j,self.num_rounds,lr,m))
        elif changeTime=='after_given_rounds':
            for i in range(0,len(self.change_lr_m)):
                j=min(j+self.change_lr_m[i][0],self.num_rounds)
                params.append((start,j,lr,m))
                lr=change(lr,self.change_lr_m[i][1])
                m=change(m,self.change_lr_m[i][2]) if len(self.change_lr_m[i])==3 else m
                start=j
            params.append((j,self.num_rounds,lr,m))
        elif changeTime=='cyclical':
            while start<self.num_rounds:
                for i in range(0,len(self.change_lr_m)):
                    j=min(start+self.change_lr_m[i][0],self.num_rounds)
                    params.append((start,j,lr,m))
                    lr=change(lr,self.change_lr_m[i][1])
                    m=change(m,self.change_lr_m[i][2]) if len(self.change_lr_m[i])==3 else m
                    start=j
        else:
            raise NotImplementedError
        return params
        
        raise NotImplementedError
        if changeType=='recursive_abs':
            return None
        if changeType=='recursive_fact':
            return None

    

def createCentralizedServer(args,model,metrics,train_transform,test_transform,root='data/idda'):
    iddaTrain=IDDADataset(root,fileName='train.txt',transform=train_transform,client_name='Centralized server')
    iddaTestSame=IDDADataset(root,fileName='test_same_dom.txt',transform=test_transform,client_name='test_same_domain')
    iddaTestDiff=IDDADataset(root,fileName='test_diff_dom.txt',transform=test_transform,client_name='test_different_domain')
    train_clients=[Client(args=args,dataset=iddaTrain,model=model)]
    test_clients=[Client(args=args,dataset=iddaTestDiff,model=model,test_client=True),Client(args=args,dataset=iddaTestSame,model=model,test_client=True)]
    return Server(args=args,train_clients=train_clients,test_clients=test_clients,model=model,metrics=metrics)


def createServerStep2(args,model,metrics,train_transform,test_transform,root='data/idda'):
    f=open(root+'/train.json')
    clients=json.load(f)
    train_clients=[]
    for key in clients:
        train_clients.append(Client(args=args,dataset=IDDADataset(root,list_samples=clients[key],transform=train_transform,client_name=key),model=model))
    iddaTestSame=IDDADataset(root,fileName='test_same_dom.txt',transform=test_transform,client_name='IDDA same domain')
    iddaTestDiff=IDDADataset(root,fileName='test_diff_dom.txt',transform=test_transform,client_name='IDDA different domain')
    test_clients=[Client(args=args,dataset=iddaTestDiff,model=model,test_client=True),Client(args=args,dataset=iddaTestSame,model=model,test_client=True)]
    return Server(args=args,train_clients=train_clients,test_clients=test_clients,model=model,metrics=metrics)