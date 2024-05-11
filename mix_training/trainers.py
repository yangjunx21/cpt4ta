from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from typing import Optional
from copy import deepcopy
from typing import Iterator, Optional, Sized
import numpy as np

class BalancedSampler(RandomSampler):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
            
        for _ in range(self.num_samples // n):
            yield from torch.randperm(n, generator=generator).tolist()
        yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

class GA_GD_Trainer(Trainer):
    def __init__(self, pretrain_model=None, theta_GA=1.0, theta_GD=1.0, **kwargs):
        super().__init__(**kwargs)
        device = self.accelerator.device
        self.theta_GA = theta_GA
        self.theta_GD = theta_GD
        self.prev_GA_loss = 0
        
    def compute_loss(self, model, inputs, return_outputs=False):
        types = inputs.pop("loss_type")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        ga_idxs = (types == 1)
        gd_idxs = (types == 2)
        kl_idxs = (types == 0)
        
        ga_inputs = {k: v[ga_idxs] for k, v in inputs.items()}
        gd_inputs = {k: v[gd_idxs] for k, v in inputs.items()}
        gd_normal_inputs = {k: v[kl_idxs] for k, v in inputs.items()}
        
        loss, valid_num = compute_crossentropy_loss(model, inputs, mean=False)
        
        loss_ga = loss[ga_idxs].mean() * -self.theta_GA
        loss_gd = loss[gd_idxs].mean() * self.theta_GA
        loss_gd_normal = loss[kl_idxs].mean() * self.theta_GD
        
        adjusted_loss = loss_ga + loss_gd + loss_gd_normal
        
        self.log({'loss_ga_harmful': loss_ga.item(), 'loss_gd': loss_gd.item(), "loss_gd_normal": loss_gd_normal.item()})
        return (adjusted_loss, outputs) if return_outputs else adjusted_loss
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)
    
class GA_GD_GD_Trainer(Trainer):
    def __init__(self, theta_GA=1.0, theta_GD=1.0, theta_KL=1.0, eos_index=None, **kwargs):
        super().__init__(**kwargs)
        self.theta_GA = theta_GA
        self.theta_GD = theta_GD
        self.theta_KL = theta_KL
        self.eos_index = eos_index
        # self.prev_GA_loss = 0
        
    def compute_loss(self, model, inputs, return_outputs=False):
        types = inputs.pop("loss_type")
        # 0 for benign query+benign response (KL), 1 for harmful query+harmful response (GA), 2 for harmful query+benign response (GD)
        # print(f'input type:{type(inputs)}')
        
        # approach 1:
        # this could lead to problem when using with gradient checkpointing and zero2/3 (gradient reduction twice)
        # ga_idxs = (types == 1)
        # gd_idxs = (types == 2) | (types == 0)
        # ga_inputs = {k: v[ga_idxs] for k, v in deepcopy(inputs).items()}
        # gd_inputs = {k: v[gd_idxs] for k, v in deepcopy(inputs).items()}
        # loss_ga = compute_crossentropy_loss(model, ga_inputs) * -self.theta_GA
        # loss_gd = compute_crossentropy_loss(model, gd_inputs) * self.theta_GD
        # adjusted_loss = loss_ga + loss_gd
        # print('=' * 5 + 'Approach 1' + '=' * 5)
        # print(f'loss_ga: {loss_ga.item()}')
        # print(f'loss_gd: {loss_gd.item()}')
        # print('=' * 5 + 'Approach 1' + '=' * 5) 


        # approach 2:
        ga_idxs = (types == 1)
        harmful_gd_idxs = (types == 2)
        normal_gd_idxs = (types == 0)
        loss_origin, eos_loss_origin = compute_crossentropy_loss(model, inputs, mean=False, eos_token_id=self.eos_index)
        print(eos_loss_origin)
        print(ga_idxs)
        loss_ga = eos_loss_origin[ga_idxs].mean() * -self.theta_GA
        loss_harmful_gd = loss_origin[harmful_gd_idxs].mean() * self.theta_GD
        loss_normal_gd = loss_origin[normal_gd_idxs].mean() * self.theta_KL
        # adjusted_loss = loss_ga + loss_harmful_gd + loss_normal_gd
        # print('=' * 5 + 'Approach 2' + '=' * 5) 
        # print(f'loss_ga: {loss_ga.item()}')
        # print(f'loss_gd: {loss_gd.item()}')
        # print('=' * 5 + 'Approach 2' + '=' * 5) 
        
        # print(f'ga_idxs:{ga_idxs}')
        # print(f'gd_idxs:{gd_idxs}')
        # print(f'inputs:{inputs["labels"][:, 40:60]}')
        # print(f'ga_inputs labels size:{ga_inputs["labels"].size()}')
        # print(f'gd_inputs labels size:{gd_inputs["labels"].size()}')
        
        if np.isnan(loss_harmful_gd.cpu().detach().numpy()):
            adjusted_loss = loss_ga + loss_normal_gd
        else:
            adjusted_loss = loss_ga + loss_harmful_gd + loss_normal_gd
        # adjusted_loss = loss_ga + loss_normal_gd
        
        self.log({'loss_ga': loss_ga.item(), 'loss_normal_gd': loss_normal_gd.item(), 'loss_harmful_gd': loss_harmful_gd.item()})

        # assert return_outputs is False
        return adjusted_loss
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return RandomSampler(self.train_dataset)
        
class GA_GD_KL_Trainer(Trainer):
    def __init__(self, pretrain_model=None, theta_GA=1.0, theta_GD=1.0, theta_KL=1.0, **kwargs):
        super().__init__(**kwargs)
        device = self.accelerator.device
        pretrain_model.to(device)
        self.pretrain_model = pretrain_model
        self.theta_GA = theta_GA
        self.theta_GD = theta_GD
        self.theta_KL = theta_KL
        # self.prev_GA_loss = 0
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # print(inputs)
        types = inputs.pop("loss_type")
        # 0 for benign query+benign response (KL), 1 for harmful query+harmful response (GA), 2 for harmful query+benign response (GD)
        ga_idxs = (types == 1)
        gd_idxs = (types == 2)
        kl_idxs = (types == 0)
        
        # print(ga_idxs.sum(), gd_idxs.sum(), kl_idxs.sum())
        
        # ga_inputs = {k: v[ga_idxs] for k, v in inputs.items()}
        # gd_inputs = {k: v[gd_idxs] for k, v in inputs.items()}
        kl_inputs = {k: v[kl_idxs] for k, v in inputs.items()}
        
        loss_origin, logits = compute_crossentropy_loss(model, inputs, mean=False, return_logits=True)
        
        loss_ga = loss_origin[ga_idxs].mean() * -self.theta_GA
        loss_gd = loss_origin[gd_idxs].mean() * self.theta_GD

        if kl_idxs.sum() == 0:
            loss_kl = torch.tensor(0., device=self.accelerator.device)
        else:
            loss_kl = compute_kl(self.pretrain_model, logits[kl_idxs], kl_inputs)
            loss_kl = loss_kl * self.theta_KL
        
        adjusted_loss = loss_ga + loss_gd + loss_kl
        
        self.log({'loss_ga': loss_ga.item(), 'loss_gd': loss_gd.item(), 'loss_kl': loss_kl.item()})
        
        # assert return_outputs is False
        return adjusted_loss
    
    # def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    #     return SequentialSampler(self.train_dataset)
    
def compute_crossentropy_loss(model, batch, mean=True, return_logits=False, eos_token_id=None):
    outputs = model(**batch)
    # print(outputs.logits.size())
    
    logits = outputs.logits
    labels = batch.get("labels")
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
    )
    valid_counts = (shift_labels != -100).sum(dim=-1).float()

    loss = loss.view(shift_logits.size(0), -1)

    loss = loss.sum(dim=-1) / valid_counts
    
    if(eos_token_id):
        
        eos_mask = (shift_labels != eos_token_id)
        # eos_mask_unsqueezed = eos_mask.unsqueeze(-1)
        # eos_mask_broadcasted = eos_mask_unsqueezed.expand(-1, -1, shift_logits.size(-1)) 
        
        # shift_logits = shift_logits[eos_mask_broadcasted].reshape(logits.size(0), logits.size(1)-2, logits.size(2)).contiguous()
        # shift_labels = shift_labels[eos_mask].reshape(eos_mask.size(0), eos_mask.size(1)-1).contiguous()
        
        # print(shift_logits.size())
        # print(shift_labels.size())
        
        eos_loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )
        
        # print(eos_mask[0].sum(), eos_mask[0].size(0))
        
        valid_counts = ((shift_labels != -100).sum(dim=-1).float() + (shift_labels != eos_token_id).sum(dim=-1).float()) / 2

        eos_loss = eos_loss.view(shift_logits.size(0), -1)

        eos_loss = (eos_loss * eos_mask).sum(dim=-1) / valid_counts
        
        # print(loss.size(), eos_loss.size())
        return loss, eos_loss
        
    if not return_logits:
        if mean:
            return loss.mean()
        else:
            return loss
    else:
        if mean:
            return loss.mean(), logits
        else:
            return loss, logits

def compute_kl(pretrained_model, logits, batch):
    """
    Compute *forward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        logits: The logits given by current unlearning model.
        batch: A batch of normal data.

    Returns:
       The KL loss.
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            input_ids=input_ids,
            labels=labels,
        )
    
    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(logits, -1)
    
    token_count = (labels != -100).sum()

    loss = -(prob_p * torch.log(prob_q + 1e-12)).masked_fill((labels == -100).unsqueeze(-1), 0).sum() / token_count

    return loss