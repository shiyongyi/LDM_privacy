import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
import random
import numpy as np
import torch

def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'results')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))
  
    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()

if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234) 
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train') # Training: 'train'; Testing: 'test'
    parser.add_argument('--load_mode', type=int, default=0)
    
    parser.add_argument('--saved_path', type=str, default=os.path.join('..', '..', 'data', 'ori_ldct')) # orginal: ori_ldct; synthetic: syn_ldct; Test: test_ldct
    
    parser.add_argument('--save_path', type=str, default=os.path.join('.','save'))
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--result_fig', type=bool, default=True)

    parser.add_argument('--norm_range_min', type=float, default=0)
    parser.add_argument('--norm_range_max', type=float, default=1)
    parser.add_argument('--trunc_min', type=float, default=0)
    parser.add_argument('--trunc_max', type=float, default=1)

    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--test_iters', type=str, default=os.path.join('..', '..', 'weights', 'RED-CNN', 'ori', 'last.ckpt')) # pre-trained checkpoint 

    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    main(args)
