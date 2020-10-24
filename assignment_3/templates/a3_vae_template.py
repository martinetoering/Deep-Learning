import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image

from datasets.bmnist import bmnist

import os
import csv
import scipy.stats as stats

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_dim),
            # Using tanh like specified in Kingma 2013
            nn.Tanh()
            )
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.std = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        out = self.encoder(input)
        mean, std = self.mean(out), self.std(out)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            # Using tanh like specified in Kingma 2013
            nn.Tanh(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
            )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.decoder(input)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        average_negative_elbo = None
        mean, std_out = self.encoder(input)
        # Make positive 
        std = torch.exp(std_out) 
        # Sampling Reparamatrization trick
        epsilon = torch.randn_like(std) # Epsilon from normal 0,1
        z = epsilon * std + mean
        out = self.decoder(z)

        l_recon = -1 * torch.sum(input*torch.log(out) + (1-input)*torch.log(1-out), dim=1)
        l_reg = -0.5 * torch.sum(std_out - std**2 - mean**2 + 1, dim=1)
        l_recon = torch.mean(l_recon)
        l_reg = torch.mean(l_reg)
        average_negative_elbo = l_recon + l_reg
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        z = torch.randn((n_samples, self.z_dim)).to(ARGS.device)
        im_means = self.decoder(z)
    
        probabilities = torch.rand(im_means.shape)
        sampled_ims = (probabilities < im_means.cpu())
        sampled_ims = sampled_ims.float()
        # sampled_ims = torch.bernoulli(im_means)
        return sampled_ims, im_means

    def manifold(self, n_samples, device):
        grid = stats.norm.ppf(torch.linspace(0.05, 0.95, n_samples))
        z = torch.stack([torch.Tensor([x, y]) for x in grid for y in grid])
        out = self.decoder(z.to(device)).view(-1, 1, 28, 28)
        return out

def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = []

    for i, inputs in enumerate(data):
        inputs = inputs.to(ARGS.device) # TODO

        inputs = inputs.view(inputs.size(0), 784)

        epoch_elbo = model(inputs)

        if model.training:

            optimizer.zero_grad()
            epoch_elbo.backward()
            optimizer.step()

        average_epoch_elbo.append(epoch_elbo.cpu().item())

    average_epoch_elbo = sum(average_epoch_elbo) / len(average_epoch_elbo)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)

def sample(file_info, results_folder, n_samples, model):
    sampled_ims, im_means = model.sample(n_samples=n_samples)
    sampled_ims = sampled_ims.view(-1, 1, 28, 28)
    im_means = im_means.view(-1, 1, 28, 28)

    grid = make_grid(sampled_ims.cpu(), nrow=4)
    grid = grid.permute(1, 2, 0)
    file_name = os.path.join(results_folder, "sampled_ims_{}.png").format(file_info)
    plt.imsave(file_name, grid.detach().numpy())

    grid = make_grid(im_means.cpu(), nrow=4)
    grid = grid.permute(1, 2, 0)
    file_name = os.path.join(results_folder, "im_means_{}.png").format(file_info)
    plt.imsave(file_name, grid.detach().numpy())

def manifold(results_folder, n_samples, model, device):
    # percent point function (ppf) inverse normal CDF 
    means = model.manifold(n_samples, device)
    file_name = os.path.join(results_folder, "manifold.png")
    grid = make_grid(means.cpu(), nrow=20)
    grid = grid.permute(1, 2, 0)
    plt.imsave(file_name, grid.detach().cpu().numpy())

def main():
    data = bmnist()[:2]  # ignore test split
    device = ARGS.device
    print("zdim:", ARGS.zdim)
    model = VAE(z_dim=ARGS.zdim).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    results_folder = "vae"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print("Created results dir: {}".format(results_folder))
    epoch_log_file = os.path.join(results_folder, "{}_train_dim{}.log".format(results_folder, ARGS.zdim))
    epoch_log_file = open(epoch_log_file, "w+")
    epoch_logger = csv.writer(epoch_log_file, delimiter='\t')
    epoch_logger.writerow(["epoch", "train_elbo", "val_elbo"])

    train_curve, val_curve = [], []
    # Sample before training
    n_samples = 16
    sample("before", results_folder, n_samples, model)

    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")


        epoch_logger.writerow([epoch, train_elbo, val_elbo])
        epoch_log_file.flush()

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        # --------------------------------------------------------------------

        # Sample each epoch
        sample("{}".format(epoch), results_folder, n_samples, model)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    if ARGS.zdim == 2:
        print("manifold")
        manifold_samples = 20
        manifold(results_folder, manifold_samples, model, device)


    save_elbo_plot(train_curve, val_curve, os.path.join(results_folder, 'elbo.pdf'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--device', type=str, default="cuda:0", 
                        help="Training device 'cpu' or 'cuda:0'")

    ARGS = parser.parse_args()

    main()
