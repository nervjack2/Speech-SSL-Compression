import os 
import torch
from tqdm import tqdm 


class MelHuBERTTools():
    def __init__(self, args, runner_config, upstream_config, upstream):
        self.args = args
        self.runner_config = runner_config
        self.upstream_config = upstream_config
        self.upstream = upstream

        self.save_every_x_epochs = self.runner_config['runner'].get('save_every_x_epochs')
        assert self.save_every_x_epochs, 'Must specify an integer for save_every_x_epochs to save model'

    def save_model(self, optimizer, global_step, num_epoch=-1, name=None):
        if global_step == 0:
            return 
        all_states = {
            'Optimizer': optimizer.state_dict(),
            'Step': global_step,
            'Args': self.args,
            'Runner': self.runner_config,
        }
        all_states = self.upstream.add_state_to_save(all_states)

        if not name:
            name = f'checkpoint-epoch-{num_epoch}.ckpt'
        save_path = os.path.join(self.args.expdir, name)
        tqdm.write(f'[MelHuBERT] - Save the checkpoint to: {save_path}')
        torch.save(all_states, save_path)