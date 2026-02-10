import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_cc(y: torch.Tensor, hm: torch.Tensor):
    """
    Compute Correlation Coefficient between predicted and ground truth saliency maps.
    
    For batched inputs (B, 1, H, W), computes CC per sample and returns mean.
    For single sample inputs (1, 1, H, W) or (H, W), computes CC over the entire tensor.
    """
    # Flatten spatial dimensions but keep batch dimension
    if y.dim() == 4:  # (B, 1, H, W) or (B, C, H, W)
        # Compute CC per sample in batch
        y_flat = y.view(y.shape[0], -1)  # (B, H*W)
        hm_flat = hm.view(hm.shape[0], -1)  # (B, H*W)
        
        # Compute mean per sample
        y_mean = y_flat.mean(dim=1, keepdim=True)  # (B, 1)
        hm_mean = hm_flat.mean(dim=1, keepdim=True)  # (B, 1)
        
        # Center the data
        vy = y_flat - y_mean  # (B, H*W)
        vhm = hm_flat - hm_mean  # (B, H*W)
        
        # Compute CC per sample
        numerator = (vy * vhm).sum(dim=1)  # (B,)
        denom_y = torch.sqrt((vy ** 2).sum(dim=1))  # (B,)
        denom_hm = torch.sqrt((vhm ** 2).sum(dim=1))  # (B,)
        
        # Avoid division by zero
        denominator = denom_y * denom_hm
        cc_per_sample = numerator / (denominator + 1e-10)
        cc_per_sample = torch.where(denominator > 1e-10, cc_per_sample, torch.zeros_like(cc_per_sample))
        
        # Return mean CC over batch
        return cc_per_sample.mean()
    else:
        # Single sample: compute over entire tensor (original behavior)
        vy = y - torch.mean(y)
        vhm = hm - torch.mean(hm)  
        if (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2))) != 0:
            cc = torch.sum(vy * vhm) / (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2)))
        else: 
            cc = torch.Tensor([0.0])
        return cc


def compute_kl(y: torch.Tensor, hm: torch.Tensor):
    """
    Compute KL divergence between predicted and ground truth saliency maps.

    This implementation is numerically stable and matches the standard
    PyTorch `KLDivLoss` usage:
      - `input` is log-probabilities
      - `target` is (non-log) probabilities
    """
    eps = 1e-10
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    # Normalize to probability distributions
    y_sum = y.view(y.shape[0], -1).sum(1, keepdim=True)
    y_distribution = y / (y_sum[:, :, None, None] + eps)

    hm_sum = hm.view(hm.shape[0], -1).sum(1, keepdim=True)
    hm_distribution = hm / (hm_sum[:, :, None, None] + eps)

    # Clamp to avoid log(0) and ensure valid probabilities
    y_distribution = y_distribution.clamp_min(eps)
    hm_distribution = hm_distribution.clamp_min(eps)

    # Input: log-probabilities, Target: probabilities
    kl = kl_loss(torch.log(y_distribution), hm_distribution)
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