import torch
import torch.nn.functional as F
import gc

def max_entropy(T, model, x):
    with torch.no_grad():
        B = x.shape[0]
        idx = torch.arange(B * T, device=x.device) % B
        x_rep = x[idx]
        model.train()
        outputs = model(x_rep)
        outputs = outputs.view(T, B, 10)
        outputs = outputs.mean(dim=0)
        individual = outputs * torch.log(outputs + 1e-10)
        entropy = -torch.sum(individual, dim=-1)

    return entropy

def bald(T, model, x):
    with torch.no_grad():
        B = x.shape[0]
        idx = torch.arange(B * T, device=x.device) % B
        x_rep = x[idx]
        model.train()
        outputs = model(x_rep)
        outputs_me = outputs.view(T, B, 10)
        outputs_me = outputs_me.mean(dim=0)
        individual = outputs_me * torch.log(outputs_me + 1e-10)
        me = -torch.sum(individual, dim=-1)

        outputs = outputs * torch.log(outputs + 1e-10)
        outputs = outputs.view(T, x.shape[0], -1)

        outputs = torch.sum(outputs, dim=-1)
        outputs = -torch.mean(outputs, dim=0)

    return me - outputs

def var_ratios(T, model, x):
    with torch.no_grad():
        x_rep = x.expand(T * x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        model.train()
        outputs = model(x_rep)
        outputs = outputs.view(T, x.shape[0], -1)

        mean_outputs = outputs.mean(dim=0)

    return 1 - torch.max(mean_outputs, dim=-1).values

def mean_std(T, model, x):
    with torch.no_grad():
        x_rep = x.expand(T * x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        model.train()
        outputs = model(x_rep)
        outputs = outputs.view(T, x.shape[0], -1).transpose(0, 1)

        std = torch.sqrt(outputs.var(dim=1))

    return torch.mean(std, dim=-1)

def random(T, model, x):
    N = x.shape[0]
    ind = torch.randint(0, N, (1,))
    return F.one_hot(ind, num_classes=N).squeeze(0).float()