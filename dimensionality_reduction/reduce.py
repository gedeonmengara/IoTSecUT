from .datasets import ContDataset
from .autoencoder import Autoencoder
from .loss import KLDMSE
from utils.logger import create_logger_autoencoder

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

class Reduce:

    def __init__(self, df_cont, data_type, H1=128, H2=24, latent_dim=3, epochs=1000, batch_size=1024, lr=1e-3, save_postfix=""):

        self.data_type = data_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = ContDataset(df_cont)
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=False)

        self.model = Autoencoder(
            D_in=df_cont.shape[1],
            H1=H1,
            H2=H2,
            latent_dim=latent_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.criterion = KLDMSE()
        self.weight_path = "weights/autoencoder"
        self.epochs = epochs
        self.save_postfix = save_postfix

    def train(self):

        self.logger, tb_dir = create_logger_autoencoder("logs", self.data_type)
        self.writer = SummaryWriter(log_dir=tb_dir)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            mse_losses = 0
            kld_losses = 0

            with tqdm(self.data_loader, unit="batch") as tloader:
                for x in tloader:
                    tloader.set_description("Epoch {}".format(epoch))

                    self.optimizer.zero_grad()
                    x = x.to(self.device)
                    recon_batch, mu, logvar = self.model(x)
                    mse_loss, kld_loss = self.criterion(recon_batch, x, mu, logvar)
                    loss = mse_loss + kld_loss

                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    mse_losses += mse_loss.item()
                    kld_losses += kld_loss.item()
                    tloader.set_postfix(loss=loss.item())

            train_loss = train_loss/len(self.data_loader)
            mse_losses = mse_losses/len(self.data_loader)
            kld_losses = kld_losses/len(self.data_loader)

            msg = 'Train Epoch : {}\t Training Loss : {}'.format(epoch, train_loss)
            self.logger.info(msg)
            self.writer.add_scalar("Loss/Total", train_loss, epoch)
            self.writer.add_scalar("Loss/MSE", mse_losses, epoch)
            self.writer.add_scalar("Loss/KLD", kld_losses, epoch)

        self.logger.info("Done")
        ckpt = {
            "model": self.model.state_dict()
        }

        # save_path = os.path.join(self.weight_path, self.data_type + ".pt")
        # save_path = os.path.join(self.weight_path, self.data_type + "_new_.pt")
        save_path = os.path.join(self.weight_path, self.data_type + f"{self.save_postfix}.pt")
        torch.save(ckpt, save_path)

    def load_model(self, save_path=None):
        if save_path == None:
            save_path = os.path.join(self.weight_path, self.data_type + ".pt")
            # save_path = os.path.join(self.weight_path, self.data_type + "_new_01.pt")
        print("Loading from ", save_path)
        ckpt = torch.load(save_path)
        self.model.load_state_dict(ckpt["model"])

    def batch_embedding(self, save_path=None):

        # Load Model
        self.load_model(save_path)

        mu_output = []
        logvar_output = []

        self.model.eval()

        with torch.no_grad():
            with tqdm(self.data_loader, unit="batch") as tloader:
                for x in tloader:
                    x = x.to(self.device)
                    recon_batch, mu, logvar = self.model(x)

                    mu_output.append(mu)
                    logvar_output.append(logvar)

        mu_result = torch.cat(mu_output, dim=0)
        logvar_result = torch.cat(logvar_output, dim=0)

        return mu_result, logvar_result

    def embedding(self, data):
        self.load_model()

        data = self.dataset.standardizer.transform(data)

        with torch.no_grad():
            data = torch.from_numpy(data).unsqueeze(0).to(self.device)
            recon_batch, mu, logvar = self.model(data)

        return mu
