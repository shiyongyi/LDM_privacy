import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import os

from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
from model.SUNet import SUNet_model
import yaml
import numpy as np

with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)


parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', default=os.path.join('..', '..', 'data', 'test_ls'), type=str, help='Input images')
parser.add_argument('--window_size', default=8, type=int, help='window size')
parser.add_argument('--result_dir', default=os.path.join('.', 'results'), type=str, help='Directory for results')
parser.add_argument('--weights',
                    default=os.path.join('..', '..', 'weights', 'SWIN', 'syn', 'last.pth'), type=str,
                    help='Path to weights')

args = parser.parse_args()


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

files = natsorted(glob(os.path.join(inp_dir, '*_input.npy')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding model architecture and weights
model = SUNet_model(opt)
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

print('restoring images......')


for file_ in files:
    img = np.load(file_)
    input_ = torch.FloatTensor(img).unsqueeze_(0).cuda()
    input_ = input_.repeat(1, 1, 1, 1)
    with torch.no_grad():
        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

    #restored = img_as_ubyte(restored[0])
    restored = restored[0]
    f = os.path.splitext(os.path.split(file_)[-1])[0]
    np.save((os.path.join(out_dir, f + '.npy')), restored)

print(f"Files saved at {out_dir}")
print('finish !')