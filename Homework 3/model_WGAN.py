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
https://arxiv.org/abs/1704.00028
https://github.com/w86763777/pytorch-gan-collections
https://github.com/eriklindernoren/PyTorch-GAN
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
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )
        self.out = nn.Linear(8192, 1)

    def forward(self, x):
        x = torch.flatten(self.main(x), 1)
        return self.out(x)


def train(batch_size=64, num_epochs=500):
    # Load data
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root="data/", download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # https://github.com/w86763777/pytorch-gan-collections
    def gradient_penalty(net_D, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
        alpha = alpha.expand(real.size())
        interpolates = alpha * real + (1 - alpha) * fake
        interpolates.requires_grad_(True)
        disc_interpolates = net_D(interpolates)
        grad = torch.autograd.grad(
            outputs=disc_interpolates, 
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True, 
            retain_graph=True
        )[0]
        grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
        loss_gp = torch.mean((grad_norm - 1) ** 2)
        return loss_gp

    # Initialize Generator and Discriminator
    netG = Generator().to(device)
    netG.apply(weights_init)
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    # Optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=[0.0,0.9])
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=[0.0,0.9])
    
    # Adaptive LR
    Num_iterations = 78200
    schedG = optim.lr_scheduler.LambdaLR(optimizerG, lambda step: 1 - step / Num_iterations)
    schedD = optim.lr_scheduler.LambdaLR(optimizerD, lambda step: 1 - step / Num_iterations)

    # Train
    iter_count = 0
    for epoch in range(1,num_epochs+1):
        g_loss = []
        d_loss = []
        for i, data in enumerate(dataloader, 0):
            iter_count += 1
            # Data batch
            X = data[0].to(device)
            batch_size = X.shape[0]
            noise = torch.randn(batch_size, 100, device=device)

            #################
            # Discriminator #
            #################
            # Real
            output_real = netD(X)
            # Fake
            fake = netG(noise)
            output_fake = netD(fake.detach())

            # Loss
            errD = -torch.mean(output_real) + torch.mean(output_fake)
            gp_errD = gradient_penalty(netD, X, fake)
            err_all = errD + 10 * gp_errD

            optimizerD.zero_grad()
            err_all.backward()
            optimizerD.step()
            d_loss.append(errD.item())

            #############
            # Generator #
            #############
            # Train Generator after 5 iterations of Discriminator
            if iter_count % 5 == 0:
                # Generate
                fake = netG(noise)
                
                # Loss
                errG = -torch.mean(netD(fake))

                optimizerG.zero_grad()
                errG.backward()
                optimizerG.step()
                g_loss.append(errG.item())

                # Update lr
                schedG.step()
                schedD.step()

        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f'% (epoch, num_epochs, sum(d_loss)/len(d_loss), sum(g_loss)/len(g_loss)))

        if epoch % 10 == 0:
            torch.save(netG.state_dict(), 'results/WGAN/models/netG_epoch_%d.pth' % (epoch))
            #torch.save(netD.state_dict(), 'results/WGAN/models/netD_epoch_%d.pth' % (epoch))