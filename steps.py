from datasets.idda import IDDADataset
from datasets.gtaV import GTAVDataset
from client import Client
from server import Server
from utils.stream_metrics import StreamSegMetrics
import json
from models.deeplabv3 import deeplabv3_mobilenetv2
import torch

class Args:
    def __init__(self,num_rounds,num_epochs,clients_per_round=1,hnm=False,lr=0.05,bs=8,wd=0,m=0.9,saveEachRounds=None,saveFolder=None,testEachRounds=None):
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
        #Functions to retrieve hyperparameters
        self.getLr=self.getHyperParamAtEpoch(self.lr)
        self.getM=self.getHyperParamAtEpoch(self.m)
        #Parameters for saving
        self.saveEachRounds=saveEachRounds
        self.saveFolder=saveFolder
        self.testEachRounds=testEachRounds
        if saveEachRounds==None:
            self.saveEachRounds=num_rounds
        if testEachRounds==None:
            self.testEachRounds=num_rounds

    def getHyperParamAtEpoch(self,param):
        if isinstance(param,float):
            return lambda x:param
        if param['type']=='polynomial':
            exponent=param['exponent'] if 'exponent' in param else 1
            start=param['from']
            end=param['to'] if 'to' in param else 0
            steps=param['steps']
            return lambda x: start+(end-start)*pow(x/steps,exponent) if x<steps else end
        if self.lr['type']=='cyclical':
            steps=param['period']/2
            min=param['min']
            max=param['max']
            return lambda x: min+(max-min)*(abs(x%(2*steps)-steps)/steps)
        raise NotImplementedError
        
    

def createServerStep1(args,train_transform,test_transform,root='data/idda',model=None):
    if model==None:
        model=deeplabv3_mobilenetv2(16)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    metrics = {
            'eval_train': StreamSegMetrics(16, 'eval_train'),
            'test_same_domain': StreamSegMetrics(16, 'test_same_domain'),
            'test_diff_domain': StreamSegMetrics(16, 'test_diff_domain')
        }
    iddaTrain=IDDADataset(root,fileName='train.txt',transform=train_transform,client_name='Centralized server')
    iddaEvalTrain=IDDADataset(root,fileName='train.txt',transform=test_transform,client_name='eval_train')
    iddaTestSame=IDDADataset(root,fileName='test_same_dom.txt',transform=test_transform,client_name='test_same_domain')
    iddaTestDiff=IDDADataset(root,fileName='test_diff_dom.txt',transform=test_transform,client_name='test_diff_domain')
    train_clients=[Client(args=args,dataset=iddaTrain,model=model)]
    test_clients=[Client(args=args,dataset=iddaEvalTrain,model=model,test_client=True),Client(args=args,dataset=iddaTestDiff,model=model,test_client=True),Client(args=args,dataset=iddaTestSame,model=model,test_client=True)]
    return Server(args=args,train_clients=train_clients,test_clients=test_clients,model=model,metrics=metrics)

def createServerStep2(args,train_transform,test_transform,root='data/idda',model=None):
    if model==None:
        model=deeplabv3_mobilenetv2(16)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    metrics = {
            'eval_train': StreamSegMetrics(16, 'eval_train'),
            'test_same_domain': StreamSegMetrics(16, 'test_same_dom'),
            'test_diff_domain': StreamSegMetrics(16, 'test_diff_domain')
        }
    f=open(root+'/train.json')
    clients=json.load(f)
    train_clients=[]
    test_clients=[Client(args=args,dataset=iddaTestDiff,model=model,test_client=True),Client(args=args,dataset=iddaTestSame,model=model,test_client=True)]
    for key in clients:
        test_clients.append(Client(args=args,dataset=IDDADataset(root,list_samples=clients[key],transform=test_transform,client_name='eval_train'),model=model))
        train_clients.append(Client(args=args,dataset=IDDADataset(root,list_samples=clients[key],transform=train_transform,client_name=key),model=model))
    iddaTestSame=IDDADataset(root,fileName='test_same_dom.txt',transform=test_transform,client_name='test_same_domain')
    iddaTestDiff=IDDADataset(root,fileName='test_diff_dom.txt',transform=test_transform,client_name='test_diff_domain')
    return Server(args=args,train_clients=train_clients,test_clients=test_clients,model=model,metrics=metrics)

def createServerStep3(args,train_transform,test_transform,rootIdda='data/idda',rootGta='data/GTA5',model=None):
    if model==None:
        model=deeplabv3_mobilenetv2(16)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    metrics = {
            'eval_train': StreamSegMetrics(16, 'eval_train'),
            'eval_target': StreamSegMetrics(16, 'eval_target'),
            'test_same_domain': StreamSegMetrics(16, 'test_same_domain'),
            'test_diff_domain': StreamSegMetrics(16, 'test_diff_domain')
        }
    gtaVtrain=GTAVDataset(rootGta,fileName='train.txt',transform=train_transform,client_name='Gta5 centralized server')
    gtaVevalTrain=GTAVDataset(rootGta,fileName='train.txt',transform=test_transform,client_name='eval_train')
    iddaTrain=IDDADataset(rootIdda,fileName='train.txt',transform=test_transform,client_name='eval_target')
    iddaTestSame=IDDADataset(rootIdda,fileName='test_same_dom.txt',transform=test_transform,client_name='test_same_domain')
    iddaTestDiff=IDDADataset(rootIdda,fileName='test_diff_dom.txt',transform=test_transform,client_name='test_diff_domain')
    train_clients=[Client(args=args,dataset=gtaVtrain,model=model)]
    test_clients=[Client(args=args,dataset=gtaVevalTrain,model=model,test_client=True),Client(args=args,dataset=iddaTrain,model=model,test_client=True),Client(args=args,dataset=iddaTestDiff,model=model,test_client=True),Client(args=args,dataset=iddaTestSame,model=model,test_client=True)]
    return Server(args=args,train_clients=train_clients,test_clients=test_clients,model=model,metrics=metrics)