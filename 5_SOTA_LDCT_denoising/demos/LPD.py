import sys
sys.path.append("..")
import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from recon.models import Learned_primal_dual
from vis_tools import Visualizer

def setup_parser(arguments, title):
    parser = argparse.ArgumentParser(description=title,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for key, val in arguments.items():
        parser.add_argument('--%s' % key,
                            type=eval(val["type"]),
                            help=val["help"],
                            default=val["default"],
                            nargs=val["nargs"] if "nargs" in val else None)
    return parser

def get_parameters(title=None):
    with open("config.json") as data_file:
        data = json.load(data_file)
    parser = setup_parser(data, title)
    parameters = parser.parse_args()
    return parameters

class net(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        options = torch.tensor([args.views, args.dets, args.width, args.height,
                                args.dImg, args.dDet, args.Ang0, args.dAng,
                                args.s2r, args.d2r, args.binshift, args.scan_type])
        self.model = Learned_primal_dual(options)
        self.epochs = args.epochs
        self.lr = args.lr
        self.is_vis_show = args.is_vis_show
        self.show_win = args.show_win
        self.is_res_save = args.is_res_save
        self.res_dir = args.res_dir        
        if self.is_vis_show:
            self.vis = Visualizer(env='LPD')

    def forward(self, x, p):
        out = self.model(x, p)
        return out

    def training_step(self, batch, batch_idx):
        x, p, y = batch
        out = self(x, p)
        loss = F.mse_loss(out, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        if self.is_vis_show:
            self.vis_show(loss.detach(), x.detach(), y.detach(), out.detach())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, p, y = batch
        out = self(x, p)
        loss = F.mse_loss(out, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, p, y, res_name = batch
        out = self(x, p)
        if self.is_res_save:
            self.res_save(out, res_name)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return [optimizer], [scheduler]     

    def show_win_norm(self, y):
        x = y.clone()
        x[x<self.show_win[0]] = self.show_win[0]
        x[x>self.show_win[1]] = self.show_win[1]
        x = (x - self.show_win[0]) / (self.show_win[1] - self.show_win[0]) * 255
        return x

    def vis_show(self, loss, x, y, out, mode='Train'):
        self.vis.plot(mode + ' Loss', loss.item())
        self.vis.img(mode + ' Ground Truth', self.show_win_norm(y).cpu())
        self.vis.img(mode + ' Input', self.show_win_norm(x).cpu())
        self.vis.img(mode + ' Result', self.show_win_norm(out).cpu())

    def res_save(self, out, res_name):
        res = out.cpu().numpy()
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)
        for i in range(res.shape[0]):
            np.save(self.res_dir + '/' + res_name[i], res[i].squeeze())

class data_loader(Dataset):
    def __init__(self, root, dose, mode):
        self.mode = mode
        self.files_x = np.array(sorted(glob.glob(os.path.join(root, dose+'_ldct', '*_input.npy'))))
        self.files_y = np.array(sorted(glob.glob(os.path.join(root, dose+'_ldct',  '*_target.npy'))))
        self.files_p = np.array(sorted(glob.glob(os.path.join(root, dose+'_sino',  '*_input.npy'))))
  
        
    def __getitem__(self, index):
        file_x = self.files_x[index]
        file_p = self.files_p[index]
        file_y = self.files_y[index]
        input_data = np.load(file_x)
        prj_data = np.load(file_p)
        label_data = np.load(file_y)
        prj_data = np.transpose(prj_data)*100
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        if self.mode == 'train' or self.mode == 'vali':
            return input_data, prj_data, label_data
        elif self.mode == 'test':
            res_name = file_x[-13:]
            return input_data, prj_data, label_data, res_name

    def __len__(self):
        return len(self.files_x)
    
if __name__ == "__main__":
    args = get_parameters()
    network = net(args)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=3, mode="min")
    trainer = pl.Trainer(gpus=args.gpu_ids if args.is_specified_gpus else args.gpus, log_every_n_steps=1, max_epochs=args.epochs, callbacks=[checkpoint_callback])
    train_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=args.cpus)
    vali_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'vali'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
    test_loader = DataLoader(data_loader(args.data_root_dir, args.dose, 'test'), batch_size=args.batch_size, shuffle=False, num_workers=args.cpus)
    trainer.fit(network, train_loader, vali_loader)
    trainer.test(network, test_loader, ckpt_path=os.path.join('..','..','weights','AdaptiveNet','ori','last.ckpt'))
    