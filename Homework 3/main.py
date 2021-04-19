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

if not os.path.exists("results"):
    os.makedirs("results")
    for model in ["DCGAN", "WGAN", "ACGAN"]:
        os.makedirs("results/{}".format(model))
        os.makedirs("results/{}/models".format(model))
        os.makedirs("results/{}/images_fake".format(model))
        os.makedirs("results/{}/images_real".format(model))

# Generate images, and save images
def generate_results(model):
    if model == "DCGAN":
        from model_DCGAN import Generator
    elif model == "WGAN":
        from model_WGAN import Generator
    else:
        from model_ACGAN import Generator
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root="data/", download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)

    netG = Generator().to(device)
    # Change 100 to different numbers to check fid at different epochs
    netG.load_state_dict(torch.load("results/{}/models/netG_epoch_100.pth".format(model)))
    fixed_noise = torch.randn(2560, 100, 1, 1, device=device)
    netG.eval()
    fake_imgs = netG(fixed_noise)

    for i, data in enumerate(dataloader):
        if i == 2560:
            break
        X = data[0].to(device)
        vutils.save_image(X,'results/{}/images_real/real_sample_{}.png'.format(model,i+1),normalize=True)
        vutils.save_image(fake_imgs[i].detach(),'results/{}/images_fake/fake_sample_{}.png'.format(model,i+1),normalize=True)


from model_DCGAN import train
train()
#generate_results("DCGAN")