import time, torch, os
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.manual_seed(1)

# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

"""
References for WGAN model
https://arxiv.org/abs/1701.07875
https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
https://github.com/w86763777/pytorch-gan-collections
https://github.com/eriklindernoren/PyTorch-GAN
"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 2, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x).squeeze(1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 5, 2, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.flatten(self.main(x), 1)
        return self.out(x).squeeze(1)


def train(batch_size=64, num_epochs=100):
    # Load data
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root="data/", download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Initialize Generator and Discriminator
    netG = Generator().to(device)
    netG.apply(weights_init)
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    # Optimizer
    optimizerD = optim.RMSprop(netD.parameters(), lr=0.0002)
    optimizerG = optim.RMSprop(netG.parameters(), lr=0.0002)

    # Train
    for epoch in range(1,num_epochs+1):
        g_loss = []
        d_loss = []
        for i, data in enumerate(dataloader, 0):
            # Data batch
            X = data[0].to(device)
            batch_size = X.shape[0]
            noise = torch.randn(batch_size, 100, 1, 1, device=device)

            # Labels
            real_lbl = torch.full((batch_size,), 1, dtype=torch.float32, device=device)
            fake_lbl = torch.full((batch_size,), 0, dtype=torch.float32, device=device)

            #################
            # Discriminator #
            #################
            # Clip weights
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)
            # Real
            netD.zero_grad()
            output_real = netD(X)

            # Fake
            fake = netG(noise)
            output_fake = netD(fake.detach())

            # Loss
            errD = -torch.mean(output_real) + torch.mean(output_fake)
            errD.backward()
            optimizerD.step()
            d_loss.append(errD.item())

            #############
            # Generator #
            #############
            if i % 5 == 0:
                netG.zero_grad()
                fake = netG(noise)
                output = netD(fake)
                # Loss
                errG = -torch.mean(netD(output))
                errG.backward()
                optimizerG.step()
                g_loss.append(errG.item())

        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f'% (epoch, num_epochs, sum(d_loss)/len(d_loss), sum(g_loss)/len(g_loss)))

        if epoch % 20 == 0:
            torch.save(netG.state_dict(), 'results/DCGAN/models/netG_epoch_%d.pth' % (epoch))
            #torch.save(netD.state_dict(), 'results/DCGAN/models/netD_epoch_%d.pth' % (epoch))