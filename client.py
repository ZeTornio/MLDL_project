import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.utils import HardNegativeMining, MeanReduction, SelfTrainingLoss

          

class Client:

    def __init__(self, args,dataset, model, teacher_model=None, test_client=False, hnm=False):
        self.args=args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.unsupervised = self.args.unsupervised
        self.teacher_model = teacher_model if self.unsupervised else None
        self.train_criterion = SelfTrainingLoss() if self.unsupervised else nn.CrossEntropyLoss(ignore_index=255, reduction='none') 
        self.reduction = HardNegativeMining() if hnm else MeanReduction()
        self.test_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none') 
        self.total_epochs=0

    def __str__(self):
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def showSample(self,index=0):
        if index>=len(self.dataset):
            print(f"Index out of bounds, the maximum is {len(self.dataset)-1}")
            return
        (image,label)=self.dataset[index]
        image = image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=torch.float32)
        self.model.eval()
        prediction=self.model(torch.unsqueeze(image,0))['out']
        _, prediction = prediction.max(dim=1)

        self.dataset.showSample(index,prediction.squeeze(0).cpu())

    def get_loss(self, outputs, labels, images):

        if self.unsupervised:
            teacher_predictions = self.teacher_model(images)['out']
            return self.train_criterion(outputs, teacher_predictions)
        else:
            return self.reduction(self.train_criterion(outputs,labels),labels)


    def run_epoch(self, cur_epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        print(f'\t\tRound epoch:{cur_epoch+1}; Total epochs of client:{self.total_epochs+1}')
        for param_group in optimizer.param_groups:
            print("\t\t\tlr:"+str(param_group['lr']))
            print("\t\t\tm:"+str(param_group['momentum']))
        
        for cur_step, (images, labels) in enumerate(self.train_loader):
            images = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=torch.float32)
            labels = labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=torch.long)
            optimizer.zero_grad()
            outputs=self.model(images)['out']
            loss = self.get_loss(outputs, labels, images)
            loss.backward()
            optimizer.step()
        self.total_epochs+=1
        
        

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        self.model.train()
        optimizer=optim.SGD(self.model.parameters(),lr=self.args.getLr(self.total_epochs),momentum=self.args.getM(self.total_epochs),weight_decay=self.args.wd)
        # TODO: check
        for epoch in range(self.args.num_epochs):
            # TODO: check
            self.run_epoch(epoch,optimizer)
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.args.getLr(self.total_epochs)
                param_group['momentum']=self.args.getM(self.total_epochs)
        return len(self.dataset),copy.deepcopy(self.model.state_dict())

    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        cumulative_loss=0.
        samples=0
        self.model.eval()
        # TODO: check
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                # TODO: check
                images = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=torch.float32)
                labels = labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=torch.long)
                outputs=self.model(images)['out']
                loss=self.reduction(self.test_criterion(outputs,labels),labels)
                samples+=images.shape[0]
                cumulative_loss += loss.item()
                self.update_metric(metric, outputs, labels)
        return cumulative_loss/samples,samples