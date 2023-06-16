import os
import torch
from torchvision import models
import numpy as np
from glob import glob

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = models.vgg19().to(device)
vgg.eval()

ori_syn = 'syn' # original dataset: 'ori'; synthetic dataset: 'syn'
path = sorted(glob(os.path.join('..', '..', 'data', ori_syn+'_dataset','*.npy'))) #path of the original or synthetic dataset

final = np.empty(shape = [0,1000])

dic_path = os.path.join('.', 'dictionary')
if not os.path.exists(dic_path):
    os.makedirs(dic_path)
    
fh = open(os.path.join(dic_path, ori_syn+'.txt'),'w')

for f in range(len(path)):
    fh.write(path[f]+'\n')
    
    img = np.load(path[f])
    img = np.array([img,img,img])
    img = img.reshape(1,3,512,512)
    img = torch.from_numpy(img)
    img = img.float()
    img = img.to(device)
    result = vgg(img)
    result = result.cpu().detach().numpy()
    final = np.append(final,result,axis=0)

fh.close()    
np.save(os.path.join(dic_path, ori_syn+'_dictionary.npy'), final) # save the dictionary for the original or synthetic dataset

