import torch
from torch.utils.data import DataLoader, TensorDataset

def evaluate(x, y, loss_fn, model):
  y_pred = model(x)
  loss = loss_fn(y, y_pred)
  return loss

def call_batchwise(fn, x, batch_size=4096, device=None):
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size * 10, shuffle=False)
    outputs = []
    for (batch,) in loader:
        if device is not None:
            batch = batch.to(device)

        out = fn(batch)
        outputs.append(out)

    return torch.cat(outputs, dim=0).to(device=device)

import torch
import math



def nll_logvar(y, gaussian_preds, moG=1):
    means = gaussian_preds[..., 0]
    log_vars = gaussian_preds[..., 1] + 1e-6

    if moG == 1:
        # Single Gaussian case
        means = means[:, 0, :]
        log_vars = log_vars[:, 0, :]

        diff = ((y - means) ** 2) / torch.exp(log_vars) \
               + torch.log(torch.tensor(2.0 * math.pi, device=y.device)) \
               + log_vars

        final_res = 0.5 * torch.sum(diff)
        return final_res

    else:
        # Mixture of Gaussians
        d = means.shape[2]
        d = torch.tensor(d, dtype=torch.float32, device=y.device)

        # Expand y to match mixture dimension
        y = y.unsqueeze(1)

        sq = (y - means) ** 2

        log_gauss = -0.5 * (
            torch.sum(sq / torch.exp(log_vars), dim=2) +
            torch.sum(log_vars, dim=2) +
            d * torch.log(torch.tensor(2.0 * math.pi, device=y.device))
        )

        K = means.shape[1]
        K = torch.tensor(K, dtype=torch.float32, device=y.device)

        log_prob = torch.logsumexp(log_gauss, dim=1) - torch.log(K)
        return log_prob


def accuracy_classification(x, y, model, device="cpu", batch_size=128):
    model.train()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            pred_labels = preds.argmax(dim=-1)
            true_labels = yb.argmax(dim=-1)

            correct += (pred_labels == true_labels).sum().item()
            total   += xb.size(0)
    return correct / total