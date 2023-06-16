import os
import sys
sys.path.append("..")

from pathlib import Path
from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import Mayo_Dataset
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.estimators import UNet
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.embedders import LabelEmbedder, TimeEmbbeding
from medical_diffusion.models.embedders.latent_embedders import VAE

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    # ------------ Load Data ----------------    
    ds = Mayo_Dataset(
        crawler_ext='npy',
        image_resize=None,
        image_crop=None,
        augment_horizontal_flip=False,
        augment_vertical_flip=False, 
        path_root=os.path.join('..', '..', 'data', 'ori_dataset'), #path of the original dataset
    )   
  
    dm = SimpleDataModule(
        ds_train = ds,
        batch_size=4, 
        # num_workers=0,
        pin_memory=True,
        # weights=ds.get_weights()
    ) 
    
    current_time = datetime.now().strftime("LDM_%Y%m%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / 'LDM'
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    # ------------ Initialize Model ------------
    # cond_embedder = None 
    cond_embedder = LabelEmbedder
    cond_embedder_kwargs = {
        'emb_dim': 1024,
        'num_classes': 2
    }
 
    time_embedder = TimeEmbbeding
    time_embedder_kwargs ={
        'emb_dim': 1024 # stable diffusion uses 4*model_channels (model_channels is about 256)
    }

    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch':4, 
        'out_ch':4, 
        'spatial_dims':2,
        'hid_chs':  [  256, 256, 512, 1024],
        'kernel_sizes':[3, 3, 3, 3],
        'strides':     [1, 2, 2, 2],
        'time_embedder':time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder':cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
        'deep_supervision': False,
        'use_res_block':True,
        'use_attention':'none',
    }

    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002, # 0.0001, 0.0015
        'beta_end': 0.02, # 0.01, 0.0195
        'schedule_strategy': 'scaled_linear'
    }
    
    # ------------ Initialize Latent Space  ------------
    latent_embedder = VAE
    latent_embedder_checkpoint = os.path.join('..', '..', 'weights', 'VAE', 'last.ckpt')
   
    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator, 
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler, 
        noise_scheduler_kwargs = noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint = latent_embedder_checkpoint,
        estimator_objective='x_T',
        estimate_variance=False, 
        use_self_conditioning=False, 
        use_ema=False,
        classifier_free_guidance_dropout=0.5, # Disable during training by setting to 0
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=1000
    )

    # -------------- Training Initialization ---------------
    to_monitor = "train/loss"  # "pl/val_loss" 
    min_max = "min"
    save_and_sample_every = 100

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=30, # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=2,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator=accelerator,
        
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        # callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every, 
        auto_lr_find=False,
        # limit_train_batches=1000,
        limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs=200,
        max_epochs=1001,
        num_sanity_val_steps=2,
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=dm)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)