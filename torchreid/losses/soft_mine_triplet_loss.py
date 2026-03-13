from __future__ import division, absolute_import
import torch
import torch.nn as nn


class SoftTripletLoss(nn.Module):
    """Triplet loss with hard mining and soft margin.
    
    Instead of using a fixed margin, it uses:
        L = log(1 + exp(dist_ap - dist_an))
    
    This provides smooth gradients and avoids the need to tune margin.
    """

    def __init__(self):
        super(SoftTripletLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size,).
        """
        n = inputs.size(0)

        # Compute pairwise squared Euclidean distance
        dist_mat = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_mat = dist_mat + dist_mat.t()
        dist_mat.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist_mat = dist_mat.clamp(min=1e-12).sqrt()  # (n, n)

        # Create mask for positive pairs (same identity)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # For each anchor, find the hardest positive and hardest negative
        dist_ap = []
        dist_an = []
        for i in range(n):
            # Hardest positive: largest distance among same ID
            pos_dists = dist_mat[i][mask[i]]
            if pos_dists.numel() > 1:  # exclude self (dist=0)
                dist_ap.append(pos_dists.max().unsqueeze(0))
            else:
                # fallback: use self (should not happen with PK sampler)
                dist_ap.append(torch.zeros(1, device=inputs.device))

            # Hardest negative: smallest distance among different IDs
            neg_dists = dist_mat[i][mask[i] == 0]
            if neg_dists.numel() > 0:
                dist_an.append(neg_dists.min().unsqueeze(0))
            else:
                # fallback: use max possible (should not happen)
                dist_an.append(torch.ones(1, device=inputs.device) * 1e6)

        dist_ap = torch.cat(dist_ap)  # (n,)
        dist_an = torch.cat(dist_an)  # (n,)

        # Soft-margin triplet loss: log(1 + exp(dist_ap - dist_an))
        loss = torch.log(1 + torch.exp(dist_ap - dist_an))
        return loss.mean()
