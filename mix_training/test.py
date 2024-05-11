from transformers import AutoModelForCausalLM
import torch

def compute_loss(model, inputs, return_outputs=False, approach=1):
    # print(inputs)
    types = inputs.pop("loss_type")
    # 0 for benign query+benign response (KL), 1 for harmful query+harmful response (GA), 2 for harmful query+benign response (GD)
    print(f'input type:{type(inputs)}')
    # approach 1:
    if approach == 1:
        ga_idxs = (types == 1)
        gd_idxs = (types == 2) | (types == 0)
        ga_inputs = {k: v[ga_idxs].contiguous() for k, v in inputs.items()}
        gd_inputs = {k: v[gd_idxs].contiguous() for k, v in inputs.items()}
        loss_ga = compute_crossentropy_loss(model, ga_inputs) * -1.0
        loss_gd = compute_crossentropy_loss(model, gd_inputs) * 1.0
        adjusted_loss = loss_ga + loss_gd
        print('=' * 5 + 'Approach 1' + '=' * 5)
        print(f'loss_ga: {loss_ga.item()}')
        print(f'loss_gd: {loss_gd.item()}')
        print('=' * 5 + 'Approach 1' + '=' * 5) 

    else:
        # approach 2:
        ga_idxs = (types == 1)
        gd_idxs = (types == 2) | (types == 0)
        loss_origin = compute_crossentropy_loss(model, inputs, mean=False)
        loss_ga = loss_origin[ga_idxs].mean() * -1.0
        loss_gd = loss_origin[gd_idxs].mean() * 1.0
        adjusted_loss = loss_ga + loss_gd
        print('=' * 5 + 'Approach 2' + '=' * 5) 
        print(f'loss_ga: {loss_ga.item()}')
        print(f'loss_gd: {loss_gd.item()}')
        print('=' * 5 + 'Approach 2' + '=' * 5) 
    
    # print(f'ga_idxs:{ga_idxs}')
    # print(f'gd_idxs:{gd_idxs}')
    # print(f'inputs:{inputs["labels"][:, 40:60]}')
    # print(f'ga_inputs:{ga_inputs["labels"][:, 40:60]}')
    # print(f'gd_inputs:{gd_inputs["labels"][:, 40:60]}')
    
    adjusted_loss = loss_ga + loss_gd
    
    print({'loss_ga': loss_ga.item(), 'loss_gd': loss_gd.item()})

    # assert return_outputs is False
    return adjusted_loss

def compute_crossentropy_loss(model, batch, mean=True):
    outputs = model(**batch)
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
    if mean:
        return loss.mean()
    else:
        return loss
    
device = torch.device('cuda:7')
path = '/data/yangjunxiao/HarmUnlearn/ft_code/hf_save/ultrafeedback_600/vicuna_format_vallinasft/seed12_Llama-2-7b-ms_bs32_warm0_lineardecay_lr2e-5_maxlen2048_max2'
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, trust_remote_code=True)
model.eval().to(device)

inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 12, 11], [1, 2, 3, 4, 5, 6, 7, 19, 12, 11]]).to(device), 'labels': torch.tensor([[-100, -100, -100, 4, 5, 6, 7, 8, 9, 10], [-100, -100, 3, 4, 5, 6, 7, 8, 12, 11], [-100, -100, 3, 4, 5, 6, 7, 19, 12, 11]]).to(device), 'loss_type': torch.tensor([0, 1, 2]).to(device)}

loss = compute_loss(model, inputs, approach=2)
loss.backward()

for name, p in model.named_parameters():
    if 'layers.1.' in name:
        print(name, p.grad)
    