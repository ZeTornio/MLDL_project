from typing import Any
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr
import matplotlib.pyplot as plt
from torch import from_numpy
import json
import numpy as np
def showIDDAsample(sample):
    fig=plt.figure()
    fig.add_subplot(2,1,1)
    plt.imshow(sample[0].permute(1,2,0))
    plt.axis('off')
    fig.add_subplot(2,1,2)
    plt.imshow(sample[1],vmax=16)
    plt.axis('off')


class_map={
    1: 13,  # ego_vehicle : vehicle
   7: 0,   # road
   8: 1,   # sidewalk
   11: 2,  # building
   12: 3,  # wall
   13: 4,  # fence
   17: 5,  # pole
   18: 5,  # poleGroup: pole
   19: 6,  # traffic light
   20: 7,  # traffic sign
   21: 8,  # vegetation
   22: 9,  # terrain
   23: 10,  # sky
   24: 11,  # person
   25: 12,  # rider
   26: 13,  # car : vehicle
   27: 13,  # truck : vehicle
   28: 13,  # bus : vehicle
   32: 14,  # motorcycle
   33: 15,  # bicycle
}

class GTAVDataset(VisionDataset):

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
        map_classes=np.zeros((256,),dtype=np.int64)
        for i in range(256):
            map_classes[i]=class_map[i] if i in class_map else 255
        return lambda x: from_numpy(map_classes[x])

    def __getitem__(self, index: int) -> Any:
        image=Image.open(self.root+'/images/'+self.list_samples[index])
        target=Image.open(self.root+'/labels/'+self.list_samples[index])
        
        
        if self.transform:
            image,target=self.transform(image,target)
        if self.target_transform:
            target=self.target_transform(target)
        return image, target

    def __len__(self) -> int:
        return len(self.list_samples)
    
