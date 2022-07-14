import os
import yaml
import glob
import random
import argparse
from shutil import copyfile
from argparse import Namespace
import torch
import numpy as np
from runner import Runner


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--runner_config', help='The yaml file for configuring the whole experiment, except the upstream model')
    parser.add_argument('-g', '--upstream_config', help='The yaml file for the upstream model')
    parser.add_argument('-n', '--expdir', help='Save experiment at this path')
    # Options
    parser.add_argument('-e', '--past_exp', help='Resume training from a checkpoint')
    parser.add_argument('-i', '--initial_weight', help='Initialize model with a specific weight')
    parser.add_argument('--init_optimizer_from_initial_weight', action='store_true', help='Initialize optimizer from -i argument as well when set to true')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    parser.add_argument('--multi_gpu', action='store_true', help='Enables multi-GPU training')

    args = parser.parse_args()
    if args.past_exp != None and args.initial_weight != None:
        raise NotImplementedError('Do not use -e and -i at the same time. -e is for resume training.')
    if args.past_exp != None and args.expdir != None:
        raise NotImplementedError('Do not use -e and -n at the same time. -e is for resume training.')

    # Do resume training 
    if args.past_exp:
        # determine checkpoint path
        if os.path.isdir(args.past_exp):
            ckpt_pths = glob.glob(f'{args.past_exp}/states-*.ckpt')
            assert len(ckpt_pths) > 0
            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
            ckpt_pth = ckpt_pths[-1]
        else:
            ckpt_pth = args.past_exp
        
        # load checkpoint
        ckpt = torch.load(ckpt_pth, map_location='cpu')

        def update_args(old, new):
            old_dict = vars(old)
            new_dict = vars(new)
            old_dict.update(new_dict)
            return Namespace(**old_dict)

        # overwrite args and config
        args = update_args(args, ckpt['Args'])
        runner_config = ckpt['Runner']
        args.past_exp = ckpt_pth
    else:
        os.makedirs(args.expdir, exist_ok=True)
        assert args.runner_config != None or args.upstream_config != None, 'Please specify .yaml config files.'

        with open(args.runner_config, 'r') as file:
            runner_config = yaml.load(file, Loader=yaml.FullLoader)

        copyfile(args.runner_config, f'{args.expdir}/config_runner.yaml')
        copyfile(args.upstream_config, f'{args.expdir}/config_model.yaml')

    return args, runner_config

def main():
    args, runner_config = get_args()
    
     # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    runner = Runner(args, runner_config)
    runner.train()
    runner.logger.close()



if __name__ == '__main__':
    main()