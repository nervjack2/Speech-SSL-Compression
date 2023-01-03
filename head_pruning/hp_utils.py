import os 
import torch
import math
from fairseq_code import MultiheadAttention
from tqdm import tqdm
from collections import defaultdict
from dataset import MelFeatDataset
from torch.utils.data import DataLoader

def set_prune_interval(prune_interval, warm_up_steps, total_prune_steps):
    if isinstance(prune_interval, int):
        tmp = [prune_interval*i for i in range(total_prune_steps)]
        prune_interval = [warm_up_steps+p for p in tmp]        
    elif isinstance(prune_interval, list):
        prune_interval = [warm_up_steps+p for p in prune_interval]
    else:
        raise NotImplementedError

    return prune_interval

class HeadPruningTools():
    def __init__(self, args, runner_config, upstream_config, upstream):
        self.args = args
        self.runner_config = runner_config
        self.upstream_config = upstream_config
        self.upstream = upstream

        self.num_layers = len(self.upstream.model.encoder.layers)
        if self.runner_config["prune"]["metric"] == "l1":
            self.num_heads_each_step = self.num_layers
        elif self.runner_config["prune"]["metric"] == "data-driven":
            self.num_heads_each_step = self.runner_config['prune']['num_heads_each_step']
        else:
            raise NotImplementedError
        
        self.total_heads = 0
        for layer in range(self.num_layers):
            self.total_heads += self.upstream.model.encoder.layers[layer].self_attn.num_heads

        self.total_prune_step = self.runner_config["prune"]["total_steps"]
        assert self.num_heads_each_step * self.total_prune_step <= self.total_heads

        self.pruned_heads = []

    def prune_api(self):
        self.prune()
        self.total_heads -= self.num_heads_each_step
        cur_heads = 0
        for layer in range(self.num_layers):
            cur_heads += self.upstream.model.encoder.layers[layer].self_attn.num_heads
        assert cur_heads == self.total_heads
        tqdm.write(f"[Head Pruning] {self.total_heads} heads are remained")

    def prune(self):
        n_to_prune = self.num_heads_each_step 

        if self.runner_config["prune"]["metric"] == "l1":
            heads_and_score = self.get_heads_norm(self.upstream.model.encoder)
        elif self.runner_config["prune"]["metric"] == "data-driven":
            heads_and_score = self.get_head_scores_by_data_driven()
        save_path = os.path.join(self.args.expdir, f'heads_and_score_{self.total_heads}.ckpt')
        torch.save(heads_and_score, save_path)
        heads_and_score = sorted(heads_and_score, key=lambda x:x[1])
        sorted_heads = [head_and_score[0] for head_and_score in heads_and_score]
     
        if self.runner_config["prune"]["target"] == "by_whole":
            """
            Ensure we don't delete all heads in a layer
            protect the top 1 scoring head in each layer
            by filtering out the heads that need to be reserved
            """
            to_protect = {l:1  for l in range(len(self.upstream.model.encoder.layers))}
            filtered_sorted_heads = []
            for layer, head in reversed(sorted_heads):
                if layer in to_protect:
                    if to_protect[layer] > 0:
                        to_protect[layer] -= 1
                        continue
                    else:
                        to_protect.pop(layer)
                filtered_sorted_heads.insert(0, (layer, head))
            sorted_heads = filtered_sorted_heads
            # Prune the lowest scoring heads
            assert len(sorted_heads) >= n_to_prune
            to_prune = sorted_heads[:n_to_prune]
        elif self.runner_config["prune"]["target"] == "by_layer":
            # Prune one head (lowest score)from each layer
            assert len(sorted_heads) >= n_to_prune
            temp = set(i for i in range(n_to_prune))
            to_prune = []
            for layer, head in sorted_heads:
                if len(temp) == 0:
                    break
                if layer in temp:
                    to_prune.append((layer, head))
                    temp.remove(layer)

        group_to_prune = {}
        for layer, head in to_prune:
            group_to_prune[layer] = group_to_prune.get(layer, [])+[head]
        tqdm.write(f'[Head Pruning] - These heads are pruned:{group_to_prune}')
        # Update pruned heads
        self.pruned_heads.append(group_to_prune)
        for idx, layer in enumerate(self.upstream.model.encoder.layers):
            if idx in group_to_prune.keys():
                assert isinstance(layer.self_attn, MultiheadAttention)
                self.prune_layer_heads(layer.self_attn, group_to_prune[idx])

    def prune_layer_heads(self, mha, heads):
        # modify from fairseq mha.adaptive_prune_heads
        # TODO:interpolate??
        # heads: list of head index to be pruned
        num_heads = mha.num_heads
        head_dim = mha.head_dim
        new_q_weight = []
        new_q_bias = []
        new_k_weight = []
        new_k_bias = []
        new_v_weight = []
        new_v_bias = []
        new_out_proj_weight = []
        for i in range(num_heads):
            if i not in heads:
                start_idx, end_idx = i*head_dim, (i+1)*head_dim
                new_q_weight.append(
                        mha.q_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                new_q_bias.append(mha.q_proj.bias[start_idx:end_idx])

                new_k_weight.append(
                    mha.k_proj.weight[
                        start_idx:end_idx,
                    ]
                )

                new_k_bias.append(mha.k_proj.bias[start_idx:end_idx])

                new_v_weight.append(
                    mha.v_proj.weight[
                        start_idx:end_idx,
                    ]
                )
                new_v_bias.append(mha.v_proj.bias[start_idx:end_idx])

                new_out_proj_weight.append(mha.out_proj.weight[:, start_idx:end_idx])
        new_q_weight = torch.cat(new_q_weight).detach()
        new_k_weight = torch.cat(new_k_weight).detach()
        new_v_weight = torch.cat(new_v_weight).detach()
        new_out_proj_weight = torch.cat(new_out_proj_weight, dim=-1).detach()
        new_q_weight.requires_grad = True
        new_k_weight.requires_grad = True
        new_v_weight.requires_grad = True
        new_out_proj_weight.requires_grad = True

        new_q_bias = torch.cat(new_q_bias).detach()
        new_q_bias.requires_grad = True

        new_k_bias = torch.cat(new_k_bias).detach()
        new_k_bias.requires_grad = True

        new_v_bias = torch.cat(new_v_bias).detach()
        new_v_bias.requires_grad = True

        mha.q_proj.weight = torch.nn.Parameter(new_q_weight)
        mha.q_proj.bias = torch.nn.Parameter(new_q_bias)

        mha.k_proj.weight = torch.nn.Parameter(new_k_weight)
        mha.k_proj.bias = torch.nn.Parameter(new_k_bias)

        mha.v_proj.weight = torch.nn.Parameter(new_v_weight)
        mha.v_proj.bias = torch.nn.Parameter(new_v_bias)

        mha.out_proj.weight = torch.nn.Parameter(new_out_proj_weight)

        mha.num_heads = num_heads-len(heads)
        #skip embed check
        new_embed_dim = mha.head_dim * mha.num_heads
        mha.embed_dim = new_embed_dim
        mha.q_proj.out_features = mha.embed_dim
        mha.k_proj.out_features = mha.embed_dim
        mha.v_proj.out_features = mha.embed_dim
        mha.out_proj.in_features = mha.embed_dim
        #[optional] disable embed_dim check
        mha._set_skip_embed_dim_check()
        return

    def get_layer_heads_norm(self, mha, layer):
        assert isinstance(mha, MultiheadAttention)
        k_proj_heads_norm = []
        q_proj_heads_norm = []
        v_proj_heads_norm = []

        for i in range(mha.num_heads):
            start_idx = i * mha.head_dim
            end_idx = (i + 1) * mha.head_dim
            k_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        mha.k_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(mha.k_proj.bias[start_idx:end_idx])).tolist()
            )
            q_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        mha.q_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(mha.q_proj.bias[start_idx:end_idx])).tolist()
            )
            v_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        mha.v_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(mha.v_proj.bias[start_idx:end_idx])).tolist()
            )

        heads_norm = []
        for i in range(mha.num_heads):
            norm = k_proj_heads_norm[i] + q_proj_heads_norm[i] + v_proj_heads_norm[i]
            heads_norm.append(((layer, i),norm))
        return heads_norm


    def get_heads_norm(self, encoder):
        heads_norm = []
        for layer in range(self.num_layers):
            layer_heads_norm = self.get_layer_heads_norm(encoder.layers[layer].self_attn, layer)
            heads_norm.extend(layer_heads_norm)
        return heads_norm

    def get_head_scores_by_data_driven(self):
        # DATA-DRIVEN PRUNING
        # Iterate over a sub-training set to get grad norm
        # Set model train mode
        self.upstream.train()
        for layer in range(self.num_layers):
            self.upstream.model.encoder.layers[layer].self_attn._set_need_intermediate(True)
        # Prepare data
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

        data_ratio = self.runner_config["prune"]["data_ratio"]
        assert 0 < data_ratio <= 1
        total_steps = int(len(dataloader.dataset)*data_ratio)
        tqdm.write(f'\n[Head Pruning] - Iterate over {data_ratio} training set, which is equivalent to {total_steps} steps')
      
        # Set optimizer
        from torch.optim import Adam
        optimizer = Adam(self.upstream.parameters(), **self.runner_config['optimizer'])    

        # set progress bar
        pbar = tqdm(total=total_steps, dynamic_ncols=True, desc='overall')
        all_loss = 0
        records = defaultdict(list)
        prefix = f'{self.args.mode}/data-drive-prune-'
        score = []
        for layer in range(self.num_layers):
            num_heads = self.upstream.model.encoder.layers[layer].self_attn.num_heads
            score.append(torch.zeros(num_heads).to(torch.device(self.args.device)))
        
        dataloader = iter(dataloader)
        while pbar.n < pbar.total:
            data = next(dataloader)
            # try/except block for forward/backward
            try:
                if pbar.n >= pbar.total:
                    break
                global_step = pbar.n + 1
              
                loss = self.upstream(
                    data,
                    global_step=global_step,
                    log_step=self.runner_config['runner']['log_step']
                )

                if self.args.multi_gpu:
                    loss = loss.sum()
                loss.backward()

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise

            # record loss
            all_loss += loss.item()
            del loss          

            # gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.upstream.model.parameters(), self.runner_config['runner']['gradient_clipping'])
            
            bsz = self.runner_config['datarc']['train_batch_size']
            for layer in range(self.num_layers):
                mha = self.upstream.model.encoder.layers[layer].self_attn
                c = mha.context_layer_val
                cg = c.grad
                num_heads = mha.num_heads
                head_dim = mha.head_dim
                assert c.size()[0] == bsz*num_heads and c.size()[2]==head_dim
                c = c.view(bsz, num_heads, -1, head_dim)
                cg = cg.view(bsz, num_heads, -1, head_dim)
                dot = torch.einsum("bhli,bhli->bhl", [cg, c])
                score[layer] += dot.abs().sum(-1).sum(0).detach()/total_steps
           
            optimizer.zero_grad()
  
            pbar.update(1)
        
        pbar.close()
        for layer in range(self.num_layers):
            self.upstream.model.encoder.layers[layer].self_attn._set_need_intermediate(False)
        
        norm_score = None
        # normalize heads norm
        heads_and_score = []
        for layer in range(self.num_layers):
            if self.runner_config["prune"]["normalize_by_layer"] is not None:
                exponent = self.runner_config["prune"]["normalize_by_layer"]
                norm_by_layer = torch.pow(torch.pow(score[layer], exponent).sum(-1), 1/exponent)
                score[layer] = score[layer] / (norm_by_layer.unsqueeze(-1) + 1e-20)
            num_heads = self.upstream.model.encoder.layers[layer].self_attn.num_heads
            for head in range(num_heads):
                heads_and_score.append(((layer, head),score[layer][head]))
        
        return heads_and_score
    
    def save_model(self, optimizer, global_step):
        # Save previous trained pruned model:
        all_states = {
            'Optimizer': optimizer.state_dict(),
            'Step': global_step,
            'Args': self.args,
            'Runner': self.runner_config,
            'Pruned_heads': self.pruned_heads
        }
        all_states = self.upstream.add_state_to_save(all_states)

        name = f'states_prune_{self.total_heads}.ckpt'
        save_path = os.path.join(self.args.expdir, name)
        tqdm.write(f'[Head Pruning] - Save the checkpoint to: {save_path}')
        tqdm.write('[Head Pruning] - Number of parameters saved: '+str(sum(p.numel() for p in all_states['model'].values())))
        torch.save(all_states, save_path)