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

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=["DCGAN", "WGAN", "ACGAN"], required=True, help="Different Model Choices")
parser.add_argument('--evaluate', choices=["train", "generate"], required=True, help="Whether to train a model or generate images")
parser.add_argument('--sample_size', default=2560, help="Number of images to generate and to sample from training set")
args = parser.parse_args()
print(args)

if args.model == "DCGAN":
    from model_DCGAN import Generator, train
elif args.model == "WGAN":
    from model_WGAN import Generator, train
elif args.model == "ACGAN":
    from model_ACGAN import Generator, train
else:
    print("Only DCGAN, WGAN, and ACGAN models can be ran")
    exit()

# Generate images, and save images
def generate_results(model):
    if model == "DCGAN":
        fixed_noise = torch.randn(args.sample_size, 100, 1, 1, device=device)
        model_file = "netG_epoch_100.pth"
    elif model == "WGAN":
        fixed_noise = torch.randn(args.sample_size, 100, device=device)
        model_file = "netG_epoch_400.pth"
    else:
        fixed_noise = torch.randn(args.sample_size, 100, 1, 1, device=device)
        model_file = "netG_epoch_100.pth"

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
        if i == args.sample_size:
            break
        X = data[0].to(device)
        save_image(X,'results/{}/images_real/real_sample_{}.png'.format(model,i+1),normalize=True)
        save_image(fake_imgs[i].detach(),'results/{}/images_fake/fake_sample_{}.png'.format(model,i+1),normalize=True)

if args.evaluate == "train":
    train()
else:
    generate_results(args.model)