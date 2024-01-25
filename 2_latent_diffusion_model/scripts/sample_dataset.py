import sys
sys.path.append("..")
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pathlib import Path
import torch 
from medical_diffusion.models.pipelines import DiffusionPipeline
import numpy as np 
import time
import os

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ------------ Load Model ------------
device = torch.device('cuda')

pipeline = DiffusionPipeline.load_from_checkpoint(os.path.join('..', '..', 'weights', 'LDM', 'last.ckpt')) # LDM checkpoint
pipeline.to(device)

if __name__ == "__main__":
    
    syn_path = os.path.join('..', '..', 'data', 'syn_dataset')
    if not os.path.exists(syn_path):
        os.makedirs(syn_path)
    for steps in [200]:
        for name, label in  {'Mayo':1}.items(): 
            n_samples = 10 # number of samples
            sample_batch = 1
            use_ddim = True 
            cfg = 1
           
            path_out = Path(syn_path) #path of the synthetic dataset            
            path_out.mkdir(parents=True, exist_ok=True)

            # --------- Generate Samples  -------------------
            torch.manual_seed(5)
            counter = 0
            for chunk in chunks(list(range(n_samples)), sample_batch):
                
                condition = torch.tensor([label]*len(chunk), device=device) if label is not None else None 
                un_cond = None # torch.tensor([1-label]*len(chunk), device=device)  if label is not None else None # Might be None, or 1-condition or specific label 
                            
                results = pipeline.sample(len(chunk), (4, 64, 64), guidance_scale=cfg, condition=condition, un_cond=un_cond, steps=steps, use_ddim=use_ddim)
                results = results.cpu().numpy()
                
                # --------- Save result ----------------♦
                
                for image in results:
                    image = (image+1)/2
                    image = image.squeeze()
                    np.save(os.path.join(path_out/f'Syn_{counter}.npy'), image)                    
                    counter += 1
          
            torch.cuda.empty_cache()
            time.sleep(3)