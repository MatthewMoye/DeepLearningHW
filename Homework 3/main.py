import argparse, time, torch, os
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.manual_seed(1)

# Generate images, and save images
def generate_results(model):
    num_samples = 2560
    if model == "DCGAN":
        from model_DCGAN import Generator
        fixed_noise = torch.randn(num_samples, 100, 1, 1, device=device)
        model_file = "netG_epoch_100.pth"
    elif model == "WGAN":
        from model_WGAN import Generator
        fixed_noise = torch.randn(num_samples, 100, device=device)
        model_file = "netG_epoch_400.pth"
    else:
        from model_ACGAN import Generator
    if not os.path.exists("results/{}/images_fake".format(model)):
        os.makedirs("results/{}/images_fake".format(model))
        os.makedirs("results/{}/images_real".format(model))

    # Load data
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root="data/", download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)

    # Load model
    netG = Generator().to(device)
    netG.load_state_dict(torch.load("results/{}/models/{}".format(model,model_file)))
    netG.eval()
    # Generate images
    with torch.no_grad():
        fake_imgs = netG(fixed_noise)
    # Save images from CIFAR-10 and generated images to directory
    for i, data in enumerate(dataloader):
        if i == num_samples:
            break
        X = data[0].to(device)
        save_image(X,'results/{}/images_real/real_sample_{}.png'.format(model,i+1),normalize=True)
        save_image(fake_imgs[i].detach(),'results/{}/images_fake/fake_sample_{}.png'.format(model,i+1),normalize=True)

# Train model
def train(model):
    if model == "DCGAN":
        from model_DCGAN import train
    elif model == "WGAN":
        from model_WGAN import train
    else:
        from model_ACGAN import train
    train()

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=["DCGAN", "WGAN", "ACGAN"], required=True, help="Different Model Choices")
parser.add_argument('--evaluate', choices=["train", "generate"], required=True, help="Whether to train a model or generate images")
args = parser.parse_args()
print(args)

if args.evaluate == "train":
    train(args.model)
else:
    generate_results(args.model)