import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision import transforms
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr
import matplotlib.pyplot as plt
from matplotlib import colors
from torchvision.io import read_image
# To be verified. Referred to after-mapping with class_eval
#0: Road, RoadLine    #5A5A5A
#1: Sidewalk   #D3D3D3
#2: Building  #D2B48C
#3: Wall   #AA4A44
#4: Fence   #964b00
#5: Pole    #C0C0C0
#6: Traffic light     #FFD580
#7: Traffic sign       #FFFF00
#8: Vegetation      #90EE90
#9: Terrain #885F47
#10: Sky   #ADD8E6
#11: Pedestrian     #D30000
#12: Rider      #FA8072
#13: Vehicle    #004999
#14: Motorcycle #0055B3
#15: Bicycle    #0062CC
#255: Unlabeled,Other k
#cmap at the moment not used
cmap = colors.ListedColormap(['#5A5A5A','#D3D3D3','#D2B48C','#AA4A44','#964b00','#C0C0C0','#FFD580','#FFFF00','#90EE90','#885F47','#ADD8E6','#D30000','#FA8072','#004999','#0055B3','#0062CC','k'])
def showIDDAsample(sample):
    fig=plt.figure()
    fig.add_subplot(2,1,1)
    plt.imshow(sample[0].permute(1,2,0))
    plt.axis('off')
    fig.add_subplot(2,1,2)
    plt.imshow(sample[1],vmax=16)
    plt.axis('off')


class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]
convert_tensor=transforms.ToTensor()

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
            f = open(fileName, "r") 
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
    
