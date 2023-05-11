from datasets.idda import IDDADataset
from client import Client
from server import Server
import json

class Args:
    def __init__(self,num_rounds,num_epochs,clients_per_round=1,hnm=False,lr=0.05,bs=16,wd=0,m=0.9):
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