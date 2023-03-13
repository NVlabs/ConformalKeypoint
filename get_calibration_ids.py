import json
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
import tqdm
import pickle

from bop_dataset import BOPDataset

# Load dataset 
dataset_name = 'lmo'
root         = './data/bop'
num_classes  = {'lmo':8, 'lmo-org':8} 
device       = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset      = BOPDataset(root, dataset_name, split='test', return_coco=True)
dataset._set_kpts_info()

n_smps  = len(dataset)
ids     = []

for i in tqdm.tqdm(range(n_smps)):
    meta        = dataset.db[i]
    path        = meta['imgpath']
    words       = path.split('/')
    names       = words[-1].split('.')
    id          = int(names[0])
    ids.append(id)
    
ids = np.array(ids)   
print(ids)

fname = 'calibration_imgs.npy'
np.save(fname,ids)
