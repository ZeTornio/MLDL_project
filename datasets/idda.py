from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision import transforms
from torchvision.datasets import VisionDataset
import utils.ss_transforms as tr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

palette=np.array([
    [128,64,128],       #road
    [244,35,232],       #sidewalk
    [70,70,70],         #building
    [102,102,156],      #wall
    [190,153,153],      #fence
    [153,153,153],      #pole
    [250,170,30],       #light
    [220,220,0],        #sign
    [107,142,35],       #vegetation
    [152,251,152],      #terrain
    [70,130,180],       #sky
    [220,20,60],        #person
    [255,0,0],          #rider
    [0,0,142],          #vehicle
    [0,0,230],          #motorcycle
    [119,11,32],        #bycicle
    [0,0,0]] )
class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]

class IDDADataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: list[str]=[],
                 fileName:str=None,
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)

        samples_from_file=[]
        if fileName!=None:
            f = open(self.root+'/'+fileName, "r") 
            samples_from_file=[x for x in f.read().split('\n') if x != '']
        self.list_samples = list_samples+samples_from_file
        self.client_name = client_name
        self.target_transform = self.get_mapping()

    @staticmethod
    def get_mapping():
        classes = class_eval
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        image=Image.open(self.root+'/images/'+self.list_samples[index]+'.jpg')
        target=Image.open(self.root+'/labels/'+self.list_samples[index]+'.png')
        if self.transform:
            image,target=self.transform(image,target)
        if self.target_transform:
            target=self.target_transform(target)
        return image, target

    def __len__(self) -> int:
        return len(self.list_samples)
    
    def showSample(self,index=0,prediction=None):
        cMap=ListedColormap(palette/256)
        image,label=self[index]
        k=3
        if prediction==None:
            k=2
        fig=plt.figure(figsize=(10*k*1080/1920,10))
        fig.add_subplot(1,k,1)
        plt.imshow(image.permute(1,2,0))
        plt.axis('off')
        fig.add_subplot(1,k,2)
        plt.imshow(label,vmax=16,cmap=cMap,interpolation='none')
        plt.axis('off')
        if prediction==None:
            return
        fig.add_subplot(1,3,3)
        plt.imshow(prediction,vmax=16,cmap=cMap,interpolation='none')
        plt.axis('off')