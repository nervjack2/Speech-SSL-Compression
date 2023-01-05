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
from pytorch_code import prune

class Runner():
    def __init__(self, args, runner_config):
        self.args = args
        self.runner_config = runner_config
        self.logger = SummaryWriter(args.expdir)                                                     
        self.upstream_config = yaml.load(open(self.args.upstream_config, 'r'), Loader=yaml.FullLoader)

        # Mode of pre-training
        if args.mode == 'melhubert':
            print('[Runner] Mode: Pre-training MelHuBERT')
            from melhubert.pretrain_expert import MelHuBERTPretrainer 
            from melhubert.mh_utils import MelHuBERTTools
            self.melhubert = MelHuBERTPretrainer(
                self.upstream_config,
                self.args.initial_weight,
                self.args.device,
                self.args.multi_gpu).to(self.args.device)
            self.mh_tools = MelHuBERTTools(
                self.args,
                self.runner_config,
                self.upstream_config,
                self.melhubert
            )
            self.save_every_x_epochs = self.mh_tools.save_every_x_epochs
        elif args.mode == 'weight-pruning':
            print(f'[Runner] Mode: weight-pruning on MelHuBERT')
            from melhubert.pretrain_expert import MelHuBERTPretrainer
            from weight_pruning.wp_utils import WeightPruningTools, get_params_to_prune
            self.melhubert = MelHuBERTPretrainer(
                self.upstream_config,
                self.args.initial_weight,
                self.args.device,
                self.args.multi_gpu).to(self.args.device)
            self.wp_tools = WeightPruningTools(
                self.args,
                self.runner_config,
                self.upstream_config,
                self.melhubert
            )
            # Initialize the pruning mask 
            params_to_prune, _ = get_params_to_prune(self.melhubert.model)
            prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.Identity,
            )
            self.total_prune_step = self.wp_tools.n_iters
            self.prune_steps = self.wp_tools.prune_steps
            self.period = self.wp_tools.period
            assert len(self.prune_steps) == self.total_prune_step, 'The length of pruning interval should equal to the total pruning steps' 
        elif args.mode == 'head-pruning':
            print(f'[Runner] Mode: {self.runner_config["prune"]["metric"]} head-pruning on MelHuBERT')
            from melhubert.pretrain_expert import MelHuBERTPretrainer
            from head_pruning.hp_utils import HeadPruningTools, set_prune_interval
            self.melhubert = MelHuBERTPretrainer(
                self.upstream_config,
                self.args.initial_weight,
                self.args.device,
                self.args.multi_gpu).to(self.args.device)
            self.hp_tools = HeadPruningTools(
                self.args,
                self.runner_config,
                self.upstream_config,
                self.melhubert
            )
            self.total_prune_step = self.runner_config['prune']['total_steps']
            self.prune_steps = set_prune_interval(
                prune_interval=self.runner_config['prune']['interval'],
                warm_up_steps=self.runner_config['prune']['warm_up'],  
                total_prune_steps=self.runner_config['prune']['total_steps']
            )
            assert len(self.prune_steps) == self.total_prune_step, 'The length of pruning interval should equal to the total pruning steps' 
        elif args.mode == 'row-pruning':
            print(f'[Runner] Mode: row-pruning on MelHuBERT')
            from melhubert.pretrain_expert import MelHuBERTPretrainer
            from row_pruning.rp_utils import RowPruningTools, set_prune_interval
            self.melhubert = MelHuBERTPretrainer(
                self.upstream_config,
                self.args.initial_weight,
                self.args.device,
                self.args.multi_gpu).to(self.args.device)
            self.row_tools = RowPruningTools(
                self.args,
                self.runner_config,
                self.upstream_config,
                self.melhubert
            )
            self.total_prune_step = self.runner_config['prune']['total_steps']
            self.prune_steps = set_prune_interval(
                prune_interval=self.runner_config['prune']['interval'],
                warm_up_steps=self.runner_config['prune']['warm_up'],  
                total_prune_steps=self.runner_config['prune']['total_steps']
            )
            assert len(self.prune_steps) == self.total_prune_step, 'The length of pruning interval should equal to the total pruning steps' 
        elif args.mode == 'distillation':
            print(f'[Runner] Mode: distillation on MelHuBERT')
            from distillation.pretrain_expert import MelHuBERTDistiller
            from melhubert.mh_utils import MelHuBERTTools
            self.melhubert = MelHuBERTDistiller(
                self.upstream_config,
                self.args.initial_weight,
                self.args.device,
                self.args.multi_gpu).to(self.args.device)
            self.mh_tools = MelHuBERTTools(
                self.args,
                self.runner_config,
                self.upstream_config,
                self.melhubert
            )
            self.save_every_x_epochs = self.mh_tools.save_every_x_epochs
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
    
        step_per_epoch = int(total_steps//n_epochs)
        
        # Check whether the pruning steps is smaller than the total amount of training steps
        if 'pruning' in self.args.mode:
            assert max(self.prune_steps) <= total_steps, f'Pruning steps {max(self.prune_steps)} should not be larger than the total training steps {total_steps}'
     
        assert self.runner_config['runner']['total_steps'] > self.runner_config['runner']['log_step']
        # Set optimizer
        optimizer = self._get_optimizer(self.melhubert)
        # set progress bar
        pbar = tqdm(total=self.runner_config['runner']['total_steps'], dynamic_ncols=True, desc='overall')

        all_loss = 0
        batch_loss = 0
        global_step = 0
        backward_steps = 0
        prefix = f'{self.args.mode}/train-'

        while pbar.n < pbar.total:
            for data in tqdm(dataloader, dynamic_ncols=True, desc='train'):
                save_or_not = (backward_steps % gradient_accumulate_steps == 0)
                if self.args.mode in ['melhubert', 'distillation']:
                    # Save model for every x epochs in MelHuBERT pre-training mode
                    if (global_step % int(self.save_every_x_epochs * step_per_epoch) == 0) and save_or_not:
                        num_epoch = global_step // step_per_epoch
                        self.mh_tools.save_model(optimizer, global_step, num_epoch)
                elif self.args.mode == 'weight-pruning':
                    if (global_step in self.prune_steps):
                        # Weight pruning
                        state = self.wp_tools.prune_api(optimizer, pbar.n, pbar.total, save_or_not)
                        if state == "not-converge":
                            pbar.total += self.period
                            self.prune_steps.append(max(self.prune_steps)+self.period)
                elif self.args.mode  == 'head-pruning':
                    if (global_step in self.prune_steps):
                        # Save model before pruning
                        if save_or_not:
                            self.hp_tools.save_model(optimizer, global_step)
                        # Head pruning
                        self.hp_tools.prune_api()       
                        # Redefine optimizer 
                        optimizer = self._get_optimizer(self.melhubert)
                elif self.args.mode  == 'row-pruning':
                    if (global_step in self.prune_steps):
                        # Save model before pruning
                        if save_or_not:
                            self.row_tools.save_model(optimizer, global_step)
                        # Row pruning
                        self.row_tools.prune_api()       
                        # Redefine optimizer 
                        optimizer = self._get_optimizer(self.melhubert)
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
                        tqdm.write(f'[Runner] - CUDA out of memory at step {global_step}')
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # Record loss
                loss_value = loss.item()
                all_loss += loss_value
                batch_loss += loss_value
                del loss
                
                # Whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                if self.args.mode == 'weight-pruning':
                    # Calculating smooth loss to exam converging during weight pruning
                    self.wp_tools.update_smooth_loss(batch_loss)
                    self.wp_tools.update_target_smooth_loss(global_step)
                    batch_loss = 0        
              
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.melhubert.model.parameters(), self.runner_config['runner']['gradient_clipping'])
                if math.isnan(grad_norm):
                    tqdm.write(f'[Runner] - Error : grad norm is NaN at global step {global_step}')
                elif not math.isnan(grad_norm):
                    optimizer.step()

                optimizer.zero_grad()

                # Logging
                if global_step % self.runner_config['runner']['log_step'] == 0 or pbar.n == pbar.total -1:
                    # Log lossx
                    if global_step % self.runner_config['runner']['log_step'] == 0:
                        all_loss /= self.runner_config['runner']['log_step']
                    else:
                        all_loss /= (global_step % self.runner_config['runner']['log_step'])
                    # print(all_loss)
                    # if global_step == 10:
                    #     exit(0)
                    self.logger.add_scalar(f'{prefix}loss', all_loss, global_step=global_step)

                    all_loss = 0
                    # Log norm
                    self.logger.add_scalar(f'{prefix}gradient norm', grad_norm, global_step=global_step)

                pbar.update(1)

        pbar.close()