import torch
from torch import nn

class KLDMSE(nn.Module):

    def __init__(self):
        super(KLDMSE, self).__init__()

        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        mse_loss = self.mse(x_recon, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return mse_loss, kld_loss