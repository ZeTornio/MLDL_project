import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision import transforms
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr
import matplotlib.pyplot as plt

def showIDDA(sample):
    #fig,(ax1,ax2)=plt.subplots(2)
    #ax1=plt.imshow(sample[0].permute(1, 2, 0)) #tensors in pytorch are channel first, need reshape
    #ax2=plt.imshow(sample[1])
    fig=plt.figure()
    fig.add_subplot(2,1,1)
    plt.imshow(sample[0].permute(1, 2, 0))
    plt.axis('off')
    fig.add_subplot(2,1,2)
    plt.imshow(sample[1])
    plt.axis('off')


class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]
convert_tensor=transforms.ToTensor()

class IDDADataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: [str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
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
        # TODO: missing code here!
        image=Image.open(self.root+'/images/'+self.list_samples[index]+'.jpg')
        image=convert_tensor(image)
        #to have pixels first instead of channel first:
        # image=from_numpy(np.asarray(image).copy())
        target=Image.open(self.root+'/labels/'+self.list_samples[index]+'.png')
        #TODO: capire perché,se e come usare il target_transform
        #target=self.target_transform(np.asarray(target))
        return image, target

    def __len__(self) -> int:
        return len(self.list_samples)
