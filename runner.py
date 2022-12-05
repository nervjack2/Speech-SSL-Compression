"""
    Training interface of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/blob/master/s3prl/pretrain/runner.py)
    Reference author: Andy T. Liu (https://github.com/andi611)
"""
import os
import math
import glob
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset import MelFeatDataset

class Runner():
    def __init__(self, args, runner_config):
        self.args = args
        self.runner_config = runner_config
        self.logger = SummaryWriter(args.expdir)                                                     
        self.upstream_config = yaml.load(open(self.args.upstream_config, 'r'), Loader=yaml.FullLoader)

        if args.mode == 'melhubert':
            from melhubert.pretrain_expert import MelHuBERTPretrainer 
            self.melhubert = MelHuBERTPretrainer(
                self.upstream_config,
                self.args.initial_weight,
                self.args.device,
                self.args.multi_gpu).to(self.args.device)
        else:
            print('We do not support this mode currently.')

    def _get_optimizer(self, model):
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), **self.runner_config['optimizer'])    

        if self.args.init_optimizer_from_initial_weight:
            all_states = torch.load(self.args.initial_weight, map_location="cpu")
            init_optimizer = all_states["Optimizer"]
            try:
                optimizer.load_state_dict(init_optimizer)
                print(f'[Runner] Load initilization optimizer weight from {self.args.initial_weight}')
            except:
                raise NotImplementedError('Could not load the initilization weight of optimizer')

        return optimizer

    def _get_dataloader(self,):
        dataset = MelFeatDataset(
            self.upstream_config['task'],
            self.runner_config['datarc']['train_batch_size'],
            self.runner_config['datarc']['sets'],
            self.runner_config['datarc']['max_timestep'],
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=1, # for bucketing
            shuffle=True, 
            num_workers=self.runner_config['datarc']['num_workers'],
            drop_last=False, 
            pin_memory=True, 
            collate_fn=dataset.collate_fn
        )
        return dataloader

    def train(self):
        # Set model train mode
        self.melhubert.train()
        # Prepare data
        gradient_accumulate_steps = self.runner_config['runner']['gradient_accumulate_steps']
        print('[Runner] - Accumulated batch size:', 
              self.runner_config['datarc']['train_batch_size'] * gradient_accumulate_steps)
        # Get dataloader
        dataloader = self._get_dataloader()
        # Convert between pre-training epochs and total steps
        n_epochs = self.runner_config['runner']['n_epochs']
        if n_epochs > 0: 
            total_steps = int(n_epochs * len(dataloader.dataset) / gradient_accumulate_steps)
            self.runner_config['runner']['total_steps'] = total_steps
            print(f'[Runner] - Training for {n_epochs} epochs, which is equivalent to {total_steps} steps')
        else:
            total_steps = self.runner_config['runner']['total_steps']
            n_epochs = int(total_steps * gradient_accumulate_steps / len(dataloader.dataset))
            print(f'[Runner] - Training for {total_steps} steps, which is approximately {n_epochs} epochs')

        # Save checkpoint for every n epochs
        save_every_x_epochs = int(self.runner_config['runner'].get('save_every_x_epochs', -1))
        if save_every_x_epochs != -1:
            assert n_epochs > 0, 'Requiring to save model per epoch, while number of epochs is lower than 1'
            step_per_epoch = int(total_steps//n_epochs)

        assert self.runner_config['runner']['total_steps'] > self.runner_config['runner']['log_step']
        assert self.runner_config['runner']['total_steps'] > self.runner_config['runner']['save_step']
        # Set optimizer
        optimizer = self._get_optimizer(self.melhubert)
        # set progress bar
        pbar = tqdm(total=self.runner_config['runner']['total_steps'], dynamic_ncols=True, desc='overall')

        all_loss = 0
        backward_steps = 0
        prefix = f'melhubert/train-'

        while pbar.n < pbar.total:
            for data in tqdm(dataloader, dynamic_ncols=True, desc='train'):
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                    loss = self.melhubert(
                        data,
                        global_step=global_step,
                        log_step=self.runner_config['runner']['log_step'],
                    )

                    if gradient_accumulate_steps > 1:
                        loss = loss / gradient_accumulate_steps
                    if self.args.multi_gpu:
                        loss = loss.sum()
                    loss.backward()

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # Record loss
                all_loss += loss.item()
                del loss
                
                # Whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.melhubert.model.parameters(), self.runner_config['runner']['gradient_clipping'])
                if math.isnan(grad_norm):
                    print(f'[Runner] - Error : grad norm is NaN at global step {global_step}')
                elif not math.isnan(grad_norm):
                    optimizer.step()

                optimizer.zero_grad()

                # Logging
                if global_step % self.runner_config['runner']['log_step'] == 0 or pbar.n == pbar.total -1:
                    # Log loss
                    if global_step % self.runner_config['runner']['log_step'] == 0:
                        all_loss /= self.runner_config['runner']['log_step']
                    else:
                        all_loss /= (global_step % self.runner_config['runner']['log_step'])
                    print(all_loss)
                    exit(0)
                    self.logger.add_scalar(f'{prefix}loss', all_loss, global_step=global_step)
            
                    all_loss = 0
                    # Log norm
                    self.logger.add_scalar(f'{prefix}gradient norm', grad_norm, global_step=global_step)
                
                # Save checkpoint for every n steps with a maximum keeping number
                if global_step % self.runner_config['runner']['save_step'] == 0 or pbar.n == pbar.total-1:
                    def check_ckpt_num(directory):
                        max_keep = self.runner_config['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.args.expdir)

                    all_states = {
                        'Optimizer': optimizer.state_dict(),
                        'Step': global_step,
                        'Args': self.args,
                        'Runner': self.runner_config    
                    }
                    all_states = self.melhubert.add_state_to_save(all_states)
                    
                    name = f'states-epoch-{n_epochs}.ckpt' if pbar.n == pbar.total -1 else f'states-{global_step}.ckpt'
                    save_path = os.path.join(self.args.expdir, name)
                    tqdm.write(f'[Runner] - Save the checkpoint to: {save_path}')
                    torch.save(all_states, save_path)

                # Save checkpoint for every n epochs
                if save_every_x_epochs != -1:
                    if global_step % (save_every_x_epochs*step_per_epoch) == 0:
                        all_states = {
                            'Optimizer': optimizer.state_dict(),
                            'Step': global_step,
                            'Args': self.args,
                            'Runner': self.runner_config,
                        }
                        all_states = self.melhubert.add_state_to_save(all_states)

                        epoch_idx = global_step // step_per_epoch
                        name = f'checkpoint-epoch-{epoch_idx}.ckpt'
                        save_path = os.path.join(self.args.expdir, name)
                        tqdm.write(f'[Runner] - Save the checkpoint to: {save_path}')
                        torch.save(all_states, save_path)

                pbar.update(1)

        pbar.close()