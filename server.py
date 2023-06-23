import copy
from collections import OrderedDict
from datetime import datetime
import numpy as np
import torch
import os
from utils.cluster import createClusters


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics, clusters=None):
        self.args=args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.saveName=self.args.saveFolder
        if self.saveName==None:
            time=datetime.now()
            self.saveName="Tests/"+time.strftime("%d-%m_%H-%M")
        if not os.path.exists(self.saveName):
            os.makedirs(self.saveName)

        self.unsupervised = self.args.unsupervised
        self.teacher_update = self.args.teacher_update
        if self.unsupervised:
            self.teacher_model = copy.deepcopy(model)
            self.teacher_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.teacher_model_params_dict = copy.deepcopy(self.teacher_model.state_dict())
        else:
            self.teacher_model=None

        self.clientsDistribution=np.ones(len(self.train_clients))
        if self.args.distribution=='uniform':
            self.clientsDistribution=np.random.uniform(1,self.args.distributionParam,len(self.train_clients))
        if self.args.distribution=='binomial':
            self.clientsDistribution=np.random.binomial(1,0.25,len(self.train_clients))+0.001

        self.clusters = clusters
        if self.args.clustering:
            self.submodels = self.submodels_init()
            if self.unsupervised:
                self.sub_teachers = self.submodels_init()

    def submodels_init(self):
        submodels = {}
        classifier_keys =list([k for k in self.model.state_dict().keys() if "classifier" in k])
        clusters = list(set(self.clusters.values()))
        for cluster in clusters:
            submodels[cluster] = OrderedDict()
            for key in classifier_keys:
                submodels[cluster][key] = copy.deepcopy(self.model_params_dict[key])
        return submodels

    def updateClientProb(self):
        if self.args.distribution=='binomial':
            for i in range(len(self.train_clients)):
                if self.clientsDistribution[i]<0.1:
                    self.clientsDistribution[i]=np.random.binomial(1,self.args.distributionParam,1)+0.001
                else:
                    self.clientsDistribution[i]=np.random.binomial(1,1-self.args.distributionParam,1)+0.001
    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        k= np.random.choice(self.train_clients, num_clients, replace=False,p=self.clientsDistribution/sum(self.clientsDistribution))
        self.updateClientProb()
        return k

    
    def loadModel(self,path):
        self.model.load_state_dict(torch.load(path,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        if self.unsupervised: 
            self.teacher_model.load_state_dict(torch.load(path,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            self.teacher_model_params_dict = copy.deepcopy(self.model.state_dict())
        
        if self.args.clustering:
            self.submodels = self.submodels_init()
            if self.unsupervised:
                self.sub_teachers = self.submodels_init()

    '''Merge of global state_dict and cluster-specific state_dict
       :param: state_dict of the global model and client name
       :return: client state_dict'''
    def client_state_dict(self, shared_state_dict, client_name, teacher=False):
        if not self.args.clustering:
            return shared_state_dict
        state_dict = copy.deepcopy(shared_state_dict)
        cluster = self.clusters.get(client_name)
        for key in state_dict:
            if key in self.submodels[cluster]:
                state_dict[key] = self.submodels[cluster][key] if teacher==False else self.sub_teachers[cluster][key]
        return state_dict

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        for i, c in enumerate(clients):
            # TODO: missing code here!
            print(f"\tCLIENT {i + 1}/{len(clients)}: {c}")
            client_model = self.client_state_dict(self.model_params_dict, c.name)
            c.model.load_state_dict(client_model)
            if self.unsupervised:
                client_teacher = self.client_state_dict(self.teacher_model_params_dict, c.name,teacher=True)
                c.teacher_model.load_state_dict(client_teacher)
            num_samples, update = c.train()
            updates.append((c.name, num_samples, update))
        return updates
    
    def aggregate_cluster_models(self, updates):
        
        classifier_keys =list([k for k in self.model_params_dict.keys() if "classifier" in k])
        bases={}
        total_weight={}
        for (client_name, client_samples, client_model) in updates:

            cluster = self.clusters.get(client_name)

            if bases.get(cluster) is None:
                bases[cluster] = OrderedDict()
            if total_weight.get(cluster) is None:
                total_weight[cluster] = 0

            total_weight[cluster] += client_samples
            for key, value in client_model.items():
                if key not in classifier_keys:
                    continue
                if key in bases[cluster]:
                    bases[cluster][key] += client_samples * value.type(torch.FloatTensor)
                else:
                    bases[cluster][key] = client_samples * value.type(torch.FloatTensor)

        for cluster, state_dict in bases.items():
            for key, value in state_dict.items():
                if total_weight[cluster] != 0:
                    self.submodels[cluster][key] = copy.deepcopy(value.type(torch.FloatTensor) / total_weight[cluster])
        
         

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """

        # "A state_dict (ovvero client_model) is simply a Python 
        # dictionary object that maps each layer to its parameter tensor"

        total_weight = 0
        base = OrderedDict()

        for (_, client_samples, client_model) in updates:

            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)

        averaged_sol_n = copy.deepcopy(self.model_params_dict)
        
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value / total_weight

        if self.args.clustering:
            self.aggregate_cluster_models(updates)

        return averaged_sol_n
    
    def update_models(self, updates, round):
        averaged_parameters = self.aggregate(updates)
        self.model.load_state_dict(averaged_parameters, strict=False)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        if self.unsupervised == False or self.teacher_update == None:
            return  
        if round % self.teacher_update == 0:
            self.teacher_model.load_state_dict(averaged_parameters, strict=False)
            self.techer_model_params_dict = copy.deepcopy(self.teacher_model.state_dict())
            if self.args.clustering == True:
                self.sub_teachers = copy.deepcopy(self.submodels)
        

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        results=np.zeros((int(self.args.num_rounds/self.args.testEachRounds)+(1 if self.args.num_rounds%self.args.testEachRounds!=0 else 0),len(self.metrics)+1))
        k=0
        for r in range(self.args.num_rounds):
            print(f"ROUND {r + 1}/{self.args.num_rounds}: Training {self.args.clients_per_round} Clients...")
            subset_clients = self.select_clients()
            updates = self.train_round(subset_clients)
            self.update_models(updates, r+1)
            if (r+1)%self.args.testEachRounds==0 and (r+1)!=self.args.num_rounds:
                self.eval_train(printRes=True)
                self.test(printRes=True)
                
                results[k,0]=r+1
                j=1
                for metric in self.metrics:
                    results[k,j]=self.metrics[metric].results['Mean IoU']
                    j+=1
                    #print(metric,': mIoU=',self.metrics[metric].results['Mean IoU'])
                k+=1
            if (r+1)%self.args.saveEachRounds==0 and (r+1)!=self.args.num_rounds:
                torch.save(self.model.state_dict(),self.saveName+"/round_"+str(r+1)+".pt")
                
        self.eval_train(printRes=True)
        self.test(printRes=True)
        results[k,0]=self.args.num_rounds
        j=1
        for metric in self.metrics:
            #print(metric,': mIoU=',self.metrics[metric].results['Mean IoU'])
            results[k,j]=self.metrics[metric].results['Mean IoU']
            j+=1
        np.savetxt(self.saveName+"/mIoU.csv", results, delimiter=",")
        torch.save(self.model.state_dict(),self.saveName+"/round_"+str(r+1)+".pt")
            
    def showClientSample(self,name=None,index=0):
        #currently does not support sub models
        if name==None:
            self.test_clients[0].model.load_state_dict(self.model_params_dict)
            return self.test_clients[0].showSample(index)
        for i in range(len(self.test_clients)):
            if self.test_clients[i].name==name:
                self.test_clients[i].model.load_state_dict(self.model_params_dict)
                self.test_clients[i].showSample(index)
                return
        print("Client not found")



    def eval_train(self,printRes=True):
        """
        This method handles the evaluation on the train clients
        """
        self.metrics['eval_train'].reset()
        for client in self.test_clients:
            if client.name.find('eval_train')<0:
                continue
            client_model = self.client_state_dict(self.model_params_dict, client.name.split("-")[1] if client.name.find("-")>=0 else client.name)
            client.model.load_state_dict(client_model)
            loss,samples=client.test(self.metrics['eval_train'])
        
        self.metrics['eval_train'].get_results()
        if printRes:
            print("Metric eval_train:\n"+str(self.metrics['eval_train']))
        

    def test(self,printRes=True):
        """
            This method handles the test on the test clients
        """
        for metric in self.metrics:
            if metric!='eval_train':
                self.metrics[metric].reset()
        for client in self.test_clients:
            if client.name.find('eval_train')>=0:
                continue
            client_model = self.client_state_dict(self.model_params_dict, client.name.split("-")[1] if client.name.find("-")>=0 else client.name)
            client.model.load_state_dict(client_model)
            metr=client.name
            if client.name.find('-')>0:
                metr=client.name[:client.name.find('-')]
            loss,samples=client.test(self.metrics[metr] )
        for metric in self.metrics:
            if metric!='eval_train':
                self.metrics[metric].get_results()
        if printRes:
            for metric in self.metrics:
                if metric!='eval_train':
                    print("Metric "+metric+":\n"+str(self.metrics[metric]))