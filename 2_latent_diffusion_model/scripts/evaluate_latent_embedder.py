import os
import sys
sys.path.append("..")
from pathlib import Path 
import logging
from datetime import datetime
from tqdm import tqdm

import numpy as np 
import torch

from torch.utils.data.dataloader import DataLoader
from measure import compute_measure
from torch.utils.data import Subset
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from medical_diffusion.models.embedders.latent_embedders import VAE
from medical_diffusion.data.datasets import Mayo_Dataset

# ----------------Settings --------------
batch_size = 1
max_samples = None # set to None for all 
target_class = None # None for no specific class 

path_out = Path.cwd()/'results'/'Mayo'/'metrics'
path_out.mkdir(parents=True, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out/f'metrics_{current_time}.log', 'w'))

# -------------- Helpers ---------------------
pil2torch = lambda x: torch.as_tensor(np.array(x)).moveaxis(-1, 0) # In contrast to ToTensor(), this will not cast 0-255 to 0-1 and destroy uint8 (required later)

# ---------------- Dataset/Dataloader ----------------
ds_real = Mayo_Dataset( #  512x512
      
    crawler_ext='npy',
    augment_horizontal_flip=False,
    augment_vertical_flip=False,    
    path_root=os.path.join('..', '..', 'data', 'ori_dataset') # path of the original dataset
)

# --------- Select specific class ------------
if target_class is not None:
    ds_real = Subset(ds_real, [i for i in range(len(ds_real)) if ds_real.samples[i][1] == ds_real.class_to_idx[target_class]])
dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)

logger.info(f"Samples Real: {len(ds_real)}")

# --------------- Load Model ------------------
model = VAE.load_from_checkpoint(os.path.join('..', '..', 'weights', 'VAE', 'last.ckpt'))
model.to(device)


# ------------- Init Metrics ----------------------
calc_lpips = LPIPS().to(device)
cout = 0
# --------------- Start Calculation -----------------
mmssim_list, mse_list = [], []
# compute PSNR, SSIM, RMSE
ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

for real_batch in tqdm(dm_real):
    imgs_real_batch = real_batch['source'].to(device)
      
    #imgs_real_batch = tF.normalize(imgs_real_batch, 0.5, 0.5) # [0, 255] -> [-1, 1]
    with torch.no_grad():
        imgs_fake_batch = model(imgs_real_batch)[0].clamp(-1, 1)
    original = imgs_real_batch.cpu().squeeze()    
    original = (original+1.0)/2.0*0.8
    synthetic = imgs_fake_batch.cpu().squeeze()
    synthetic = (synthetic+1.0)/2.0*0.8
   
    cout = cout + 1
   
    original_result, pred_result = compute_measure(original, original, synthetic, 1.0)
    ori_psnr_avg += original_result[0]
    ori_ssim_avg += original_result[1]
    ori_rmse_avg += original_result[2]
    pred_psnr_avg += pred_result[0]
    pred_ssim_avg += pred_result[1]
    pred_rmse_avg += pred_result[2]


print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/cout, 
                                                                                      pred_ssim_avg/cout, 
                                                                                      pred_rmse_avg/cout))
   

