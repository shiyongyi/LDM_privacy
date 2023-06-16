import os

import torch
import yaml

from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time
import utils
import numpy as np
import random

from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model

import glob


## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


## Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

## Build Model
print('==> Build the model')
model_restored = SUNet_model(opt)
p_number = network_parameters(model_restored)
model_restored.cuda()

## Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join('.', 'checkpoints')
utils.mkdir(model_dir)
train_dir = os.path.join('..', '..', 'data', 'ori_ldct')

## GPU
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

## Log
log_dir = os.path.join(model_dir, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

## Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Loss
MSE_loss = nn.MSELoss()


class data_loader(Dataset):
    def __init__(self, root, mode):
        self.mode = mode
        self.files_x = np.array(sorted(glob.glob(os.path.join(train_dir, '*_input.npy'))))
        self.files_y = np.array(sorted(glob.glob(os.path.join(train_dir, '*_target.npy'))))
        
        
    def __getitem__(self, index):
        file_x = self.files_x[index]
        file_y = self.files_y[index]
        input_data = np.load(file_x)
        label_data = np.load(file_y)

        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        if self.mode == 'train' or self.mode == 'vali':
            return input_data, label_data
        elif self.mode == 'test':
            res_name = file_x[-13:]
            return input_data, label_data, res_name

    def __len__(self):
        return len(self.files_x)
    
## DataLoaders
print('==> Loading datasets')
train_loader = DataLoader(data_loader(train_dir,'train'), batch_size = 1, shuffle = True)


# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restored.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        # Forward propagation
        for param in model_restored.parameters():
            param.grad = None
        target = data[1].cuda()
        input_ = data[0].cuda()
        restored = model_restored(input_)

        # Compute loss
        #loss = Charbonnier_loss(restored, target)
        loss = MSE_loss(restored, target)

        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
# Save the last model
torch.save({'epoch': epoch,
            'state_dict': model_restored.state_dict(),
            'optimizer': optimizer.state_dict()
            }, os.path.join(model_dir, "SUNet_{:d}.pth".format(epoch)))

    
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
