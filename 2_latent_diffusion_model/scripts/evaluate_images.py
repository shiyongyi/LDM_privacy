import sys
sys.path.append("..")
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pathlib import Path 
import logging
from datetime import datetime
from tqdm import tqdm

import numpy as np 
import torch
from torch.utils.data.dataloader import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from medical_diffusion.data.datasets import Mayo_3_Dataset

# ----------------Settings --------------
batch_size = 1
max_samples = None # set to None for all 

path_out = Path.cwd()/'results'/'Mayo'/'metrics'
path_out.mkdir(parents=True, exist_ok=True)


# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out/f'metrics_{current_time}.log', 'w'))

# -------------- Helpers ---------------------
pil2torch = lambda x: torch.as_tensor(np.array(x)).moveaxis(-1, 0) # In contrast to ToTensor(), this will not cast 0-255 to 0-1 and destroy uint8 (required later)


# ---------------- Dataset/Dataloader ----------------
ds_real = Mayo_3_Dataset( #  512x512  
    crawler_ext='npy',
    augment_horizontal_flip=False,
    augment_vertical_flip=False,
    path_root=os.path.join('..', '..', 'data', 'ori_dataset') #path of the original dataset
)

ds_fake = Mayo_3_Dataset( #  512x512  
    crawler_ext='npy',
    augment_horizontal_flip=False,
    augment_vertical_flip=False,
    path_root=os.path.join('..', '..', 'data', 'syn_dataset') #path of the synthetic dataset
)

dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)
dm_fake = DataLoader(ds_fake, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)

logger.info(f"Samples Real: {len(ds_real)}")
logger.info(f"Samples Fake: {len(ds_fake)}")

# ------------- Init Metrics ----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calc_fid = FID().to(device) # requires uint8

# --------------- Start Calculation -----------------
for real_batch in tqdm(dm_real):
    temp = real_batch['source'].numpy()
    temp = np.array(temp,dtype='uint8')
    
    imgs_real_batch = torch.from_numpy(temp).to(device)
    
    # -------------- FID -------------------
    calc_fid.update(imgs_real_batch, real=True)

for fake_batch in tqdm(dm_fake):
    temp = fake_batch['source'].numpy()
    temp = np.array(temp,dtype='uint8')
    imgs_fake_batch = torch.from_numpy(temp).to(device)
    
    # -------------- FID -------------------
    calc_fid.update(imgs_fake_batch, real=False)

# -------------- Summary -------------------
fid = calc_fid.compute()
logger.info(f"FID Score: {fid}")