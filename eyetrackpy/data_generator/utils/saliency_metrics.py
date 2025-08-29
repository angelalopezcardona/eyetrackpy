import torch

def compute_cc(y: torch.Tensor, hm: torch.Tensor):
    vy = y - torch.mean(y)
    vhm = hm - torch.mean(hm)  
    if (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2))) != 0:
        cc = torch.sum(vy * vhm) / (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2)))
    else: 
        cc = torch.Tensor([0.0])
    return cc


def compute_kl(y: torch.Tensor, hm: torch.Tensor):
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    eps=1e-10
    y_sum = y.view(y.shape[0], -1).sum(1, keepdim=True)
    y_distribution = y / (y_sum[:, :, None, None] + eps)

    hm_sum = hm.view(y.shape[0], -1).sum(1, keepdim=True)
    hm_distribution = hm / (hm_sum[:, :, None, None] + eps)
    hm_distribution = hm_distribution + eps
    hm_distribution = hm_distribution / (1+eps)
    kl = kl_loss(torch.log(y_distribution), torch.log(hm_distribution))
    return kl

def compute_nss(y: torch.Tensor, fix: torch.Tensor):
    if fix.sum() != 0:
        normal_y = (y-y.mean())/y.std()
        nss = torch.sum(normal_y*fix)/fix.sum()
    else:
        nss = torch.Tensor([0.0])
    return nss