import copy
from collections import OrderedDict

import numpy as np
import torch


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.updates = []

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)
    
    def add_updates(self, num_samples, updates):
        self.updates.append((num_samples, updates))

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        for i, c in enumerate(clients):
            # TODO: missing code here!
            print(f"CLIENT {i + 1}/{len(clients)}: {c}")

            c.model.load_state_dict(self.model_params_dict)

            num_samples, update = c.train()

            updates[i] = (num_samples, update)

            # oppure
            #self.add_updates(num_samples=num_samples, update=update)

        return updates

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        # TODO: missing code here!

        # "A state_dict (ovvero client_model) is simply a Python 
        # dictionary object that maps each layer to its parameter tensor"

        total_weight = 0
        base = OrderedDict()

        for (client_samples, client_model) in self.updates:

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

        return averaged_sol_n
    
    def update_model(self, updates):
        
        averaged_parameters = self.aggregate(updates)

        self.model.load_state_dict(averaged_parameters, strict=False)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        #self.updates = []
        

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        
        for r in range(self.args.num_rounds):
            # TODO: missing code here!

            print(f"ROUND {r + 1}/{self.args.num_rounds}: Training {self.args.clients_per_round} Clients...")

            subset_clients = self.select_clients()

            updates = self.train_round(subset_clients)

            self.update_model(updates)
            
    

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        # TODO: missing code here!
        raise NotImplementedError

    def test(self):
        """
            This method handles the test on the test clients
        """
        # TODO: missing code here!
        raise NotImplementedError
