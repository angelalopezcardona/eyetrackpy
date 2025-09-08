import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_cc(y: torch.Tensor, hm: torch.Tensor):
    vy = y - torch.mean(y)
    vhm = hm - torch.mean(hm)  
    if (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2))) != 0:
        cc = torch.sum(vy * vhm) / (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2)))
    else: 
        cc = torch.Tensor([0.0])
    return cc


def compute_kl(y: torch.Tensor, hm: torch.Tensor):
    eps = 1e-10
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    # Normalizza in distribuzioni
    y_distribution = y / (y.view(y.shape[0], -1).sum(1, keepdim=True)[:, :, None, None] + eps)
    hm_distribution = hm / (hm.view(hm.shape[0], -1).sum(1, keepdim=True)[:, :, None, None] + eps)

    # Evita log(0)
    y_distribution = torch.clamp(y_distribution, min=eps)
    hm_distribution = torch.clamp(hm_distribution, min=eps)

    # KL(gt || model)
    kl = kl_loss(torch.log(y_distribution), torch.log(hm_distribution))
    return kl

def compute_nss(y: torch.Tensor, fix: torch.Tensor):
    if fix.sum() != 0:
        normal_y = (y-y.mean())/y.std()
        nss = torch.sum(normal_y*fix)/fix.sum()
    else:
        nss = torch.Tensor([0.0])
    return nss

def compute_auc(y: torch.Tensor, fix: torch.Tensor):
    """
    Compute Area Under the Curve (AUC) for saliency prediction.
    
    Args:
        y: Predicted saliency map (torch.Tensor)
        fix: Ground truth fixation map (torch.Tensor) - binary map with 1s at fixation points
    
    Returns:
        AUC score (float)
    """
    # Convert to numpy for sklearn
    y_np = y.detach().cpu().numpy().flatten()
    fix_np = fix.detach().cpu().numpy().flatten()
    
    # Ensure fixation map is binary
    fix_binary = (fix_np > 0).astype(int)
    
    # Check if we have both positive and negative samples
    if len(np.unique(fix_binary)) < 2:
        return torch.tensor(0.5, dtype=torch.float32)  # Random performance if no fixations or all fixations
    
    try:
        auc = roc_auc_score(fix_binary, y_np)
        return torch.tensor(auc, dtype=torch.float32)
    except ValueError:
        return torch.tensor(0.5, dtype=torch.float32)  # Return random performance on error

def compute_sim(y: torch.Tensor, hm: torch.Tensor):
    """
    Compute Similarity (SIM) / Histogram Intersection between predicted and ground truth saliency.
    
    Args:
        y: Predicted saliency map (torch.Tensor)
        hm: Ground truth saliency map (torch.Tensor)
    
    Returns:
        SIM score (float) - range [0, 1], higher is better
    """
    eps = 1e-10
    
    # Normalize both maps to sum to 1 (convert to probability distributions)
    y_sum = y.view(y.shape[0], -1).sum(1, keepdim=True)
    y_norm = y / (y_sum[:, :, None, None] + eps)
    
    hm_sum = hm.view(hm.shape[0], -1).sum(1, keepdim=True)
    hm_norm = hm / (hm_sum[:, :, None, None] + eps)
    
    # Compute histogram intersection (minimum of corresponding bins)
    sim = torch.sum(torch.min(y_norm, hm_norm), dim=[1, 2, 3])
    
    return sim