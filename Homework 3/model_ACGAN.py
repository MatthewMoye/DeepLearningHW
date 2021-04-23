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
References for ACGAN model
https://arxiv.org/abs/1610.09585
https://github.com/eriklindernoren/PyTorch-GAN
https://github.com/kimhc6028/acgan-pytorch
"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = nn.Linear(100, 8192)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.dense(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25)
        )
        self.s = nn.Sequential(
            nn.Linear(8192, 1),
            nn.Sigmoid()
        )
        self.c = nn.Sequential(
            nn.Linear(8192, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = torch.flatten(self.main(x), 1)
        x = torch.flatten(x, 1)
        return self.s(x).squeeze(1), self.c(x)


def train(batch_size=64, num_epochs=200):
    # Load data
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root="data/", download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Initialize Generator and Discriminator
    netG = Generator().to(device)
    netG.apply(weights_init)
    netD = Discriminator().to(device)
    netD.apply(weights_init)
    print(netG)
    print(netD)

    # Loss
    source_criterion = nn.BCELoss()
    class_criterion = nn.NLLLoss()

    # Optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Train
    for epoch in range(1,num_epochs+1):
        g_loss = []
        d_loss = []
        for i, data in enumerate(dataloader, 0):
            # Data batch
            X = data[0].to(device)
            class_labels = data[1].to(device)
            batch_size = X.shape[0]
            noise = torch.randn(batch_size, 100, device=device)
            gen_labels = torch.randint(0, 10, (batch_size,), device=device)

            # Real and Fake Labels
            real_ident = torch.full((batch_size,), 1, dtype=torch.float32, device=device)
            fake_ident = torch.full((batch_size,), 0, dtype=torch.float32, device=device)

            #############
            # Generator #
            #############
            netG.zero_grad()
            fake = netG(noise)
            valid, lbl = netD(fake)
            errG_s = source_criterion(valid, real_ident)
            errG_c = class_criterion(lbl, gen_labels)
            errG = errG_s + errG_c
            errG.backward()
            optimizerG.step()

            #################
            # Discriminator #
            #################
            netD.zero_grad()
            # Real
            real_pred, real_lbl = netD(X)
            errD_s = source_criterion(real_pred, real_ident)
            errD_c = class_criterion(real_lbl, class_labels)
            real_errD = errD_s + errD_c

            # Fake
            fake_pred, fake_lbl = netD(fake.detach())
            errD_s = source_criterion(fake_pred, fake_ident)
            errD_c = class_criterion(fake_lbl, gen_labels)
            fake_errD = errD_s + errD_c

            errD = real_errD + fake_errD            
            errD.backward()
            optimizerD.step()

            g_loss.append(errG.item())
            d_loss.append(errD.item())

        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f'% (epoch, num_epochs, sum(d_loss)/len(d_loss), sum(g_loss)/len(g_loss)))

        if epoch % 20 == 0:
            torch.save(netG.state_dict(), 'results/ACGAN/models/netG_epoch_%d.pth' % (epoch))
            #torch.save(netD.state_dict(), 'results/ACGAN/models/netD_epoch_%d.pth' % (epoch))