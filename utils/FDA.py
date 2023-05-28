from PIL import Image
import utils.ss_transforms as sstr
import numpy as np
import torch
import json

def extractStyle(image,n,size):
    if n%2==0:
        raise ValueError("You must use odd numbers >1 as windows for FDA.")
    k=int(n/2)
    im_src = Image.open(image)
    converter=sstr.Compose([sstr.Resize(size),sstr.ToTensor()])
    img=converter(im_src)
    im_fft=torch.fft.rfft2(img)
    im_ampl=im_fft.abs()
    return im_ampl[:,:k+1,:k+1], im_ampl[:,-k:,:k+1]

def extractClientsStyles(n,fileName='data/idda/train.json',folder="data/idda/images/",size=(540,960)):
    f=open(fileName)
    clients=json.load(f)
    if n%2==0:
        raise ValueError("You must use odd numbers >1 as windows for FDA. In applyStyle you can use any odd int smaller than the chosen one here, also 1.")
    pos=torch.zeros((3,int(n/2)+1,int(n/2)+1))
    neg=torch.zeros((3,int(n/2),int(n/2)+1))
    styles={}
    for client in clients:
        print("Extracting style of "+client)
        pos*=.0
        neg*=.0
        styles[client]={}
        for img in clients[client]:
            posImg,negImg=extractStyle(folder+img+'.jpg',n,size)
            pos+=posImg
            neg+=negImg
        pos/=len(clients[client])
        neg/=len(clients[client])
        styles[client]['pos']=pos.clone()
        styles[client]['neg']=neg.clone()
    return styles.copy()

def applyStyle(img,style,n):
    if n%2==0:
        raise ValueError("You must use odd numbers as windows for FDA.")
    k=int(n/2)
    im_fft=torch.fft.rfft2(img)
    im_ampl=im_fft.abs()
    im_phase=im_fft.angle()
    im_ampl[:,:k+1,:k+1]=style['pos'][:,:k+1,:k+1]
    if k>0:
        im_ampl[:,-k:,:k+1]=style['neg'][:,-k:,:k+1]
    im_rec_fft=torch.complex(torch.cos(im_phase)*im_ampl,torch.sin(im_phase)*im_ampl)
    return torch.fft.irfft2(im_rec_fft)

class applyFDAstyles(object):
    """Change the style using the FDA procedure.
    Args:
        styles (dict): the various styles to extract from.
        n (odd int): the number of pixel in low freq. to change.
    """

    def __init__(self, styles,n):
        self.styles = styles
        self.n=n

    def __call__(self, img, lbl=None):
        """
        Args:
            img (Tensor representation of Image): Image to which the style is applied.
        Returns:
            PIL Image: Randomly flipped image.
        """
        key=np.random.choice(list(self.styles.keys()), 1)
        style=self.styles[key[0]]
        if lbl is not None:
            return applyStyle(img,style,self.n), lbl
        else:
            return applyStyle(img,style,self.n)

    def __repr__(self):
        return self.__class__.__name__ + '(n={})'.format(self.n)