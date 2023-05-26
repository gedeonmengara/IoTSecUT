import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):

    def __init__(self, D_in, H1, H2, latent_dim):
        super(Encoder, self).__init__()

        self.lin1 = nn.Linear(D_in,H1)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H1)
        self.lin2 = nn.Linear(H1,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.lin3 = nn.Linear(H2,H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)
        
        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.lin_bn1(self.lin1(x)))
        x = self.relu(self.lin_bn2(self.lin2(x)))
        x = self.relu(self.lin_bn3(self.lin3(x)))

        x = F.relu(self.bn1(self.fc1(x)))

        mu = self.fc21(x)
        logvar = self.fc22(x)

        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, D_in, H1, H2, latent_dim):
        super(Decoder, self).__init__()

#         # Sampling vector
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn1 = nn.BatchNorm1d(latent_dim)
        self.fc2 = nn.Linear(latent_dim, H2)
        self.fc_bn2 = nn.BatchNorm1d(H2)

        self.lin1 = nn.Linear(H2, H2)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H2)
        self.lin2 = nn.Linear(H2, H1)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H1)
        self.lin3 = nn.Linear(H1, D_in)
        self.lin_bn3 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.relu(self.fc_bn2(self.fc2(x)))

        x = self.relu(self.lin_bn1(self.lin1(x)))
        x = self.relu(self.lin_bn2(self.lin2(x)))

        return self.lin_bn3(self.lin3(x))

class Autoencoder(nn.Module):

    def __init__(self, D_in, H1, H2, latent_dim):

        super(Autoencoder, self).__init__()

        self.enc = Encoder(D_in, H1, H2, latent_dim)
        self.dec = Decoder(D_in, H1, H2, latent_dim)

    def forward(self, x):
        mu, logvar = self.enc(x)

        if self.training:
            # print("Training")
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            z = eps.mul(std).add_(mu)

        else:
            # print("Eval")
            z = mu

        return self.dec(z), mu, logvar
