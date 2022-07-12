import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable

import csv
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.generator = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), 
            nn.Tanh() # Radford et al. 2015 DCGAN 
            )

    def forward(self, z):
        # Generate images from z
        out = self.generator(z)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.discriminator = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # Output in range [0, 1]
            )

    def forward(self, img):
        # return discriminator score for img
        out = self.discriminator(img)
        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    device = args.device

    results_folder = "gan"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print("Created results dir: {}".format(results_folder))
    log_file = os.path.join(results_folder, "{}_train_dim{}.log".format(results_folder, args.latent_dim))
    log_file = open(log_file, "w+")
    logger = csv.writer(log_file, delimiter='\t')
    logger.writerow(["epoch", "i", "loss_g", "loss_d"])

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            imgs = imgs.to(device)
            zero_targets = Variable(torch.zeros(imgs.size(0), 1, dtype=imgs.dtype), requires_grad=False).to(device)
            one_targets = Variable(torch.ones(imgs.size(0), 1, dtype=imgs.dtype), requires_grad=False).to(device)

            # Train Generator
            # ---------------
            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.size(0), args.latent_dim))))
            noise = noise.to(device)
            optimizer_G.zero_grad()
            out_G = generator(noise)
            gen_imgs = out_G.view(out_G.size(0), 1, 28, 28)
            out_D_G = discriminator(out_G)
            loss_G = -torch.mean(torch.log(out_D_G))
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            imgs = imgs.view(imgs.size(0), 784)
            optimizer_D.zero_grad()
            out_D_real = discriminator(imgs)
            loss_D_real = -torch.mean(torch.log(out_D_real))

            out_D_fake = discriminator(out_G.detach())
            loss_D_fake = -torch.mean(torch.log(1-out_D_fake))
            
            loss_D = (loss_D_real + loss_D_fake) 
            loss_D.backward()
            optimizer_D.step()

            if i % 10 == 0:
                print("epoch:", epoch, "iteration:", i)
                loss_G_item = loss_G.cpu().item()
                loss_D_item = loss_D.cpu().item()
                print("Epoch {:04d}/{:04d}, Iteration {:04d}, Batch Size = {}, "
                    "Loss generator = {:.3f}, Loss discriminator = {:.3f}, "
                    "Latent dim = {}".format(
                        epoch, args.n_epochs, i, args.batch_size, loss_G_item, 
                        loss_D_item, args.latent_dim
                ))
                logger.writerow([epoch, i, loss_G_item, loss_D_item])
                log_file.flush()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(gen_imgs[:25],
                           'gan_images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)

    if args.interpolate:
        noise_start = Variable(torch.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))))
        noise_end = Variable(torch.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))))
        n_samples = 9
        step = (noise_end-noise_start) / (n_samples - 1)
        out_samples = torch.Tensor().to(device)
        for i in range(n_samples):
            noise = noise_start + (i*step)
            noise = noise.to(device)
            sample = generator(noise).view(-1, 1, 28, 28)
            out_samples = torch.cat((out_samples, sample[:1]))

        save_image(out_samples,
               'gan_images/interpolate.png',
               nrow=9, normalize=True)


def main():
    # Create output image directory
    os.makedirs('gan_images', exist_ok=True)
    device = args.device

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),(0.5,))])),
        batch_size=args.batch_size, shuffle=True) # TODO

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Resume 
    if args.resume_path:
        checkpoint = torch.load(args.resume_path)
        generator.load_state_dict(checkpoint)
        print("model loaded from: {}".format(args.resume_path))

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type=str, default="cuda:0", 
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--interpolate', type=bool, default=False, 
                        help="If interpolating latent space should be done")
    parser.add_argument('--resume_path', type=str, 
                        help="Where to load stored model from")
    args = parser.parse_args()

    main()
