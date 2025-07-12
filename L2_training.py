from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML
from diffusion_utilities import *

## ------------------------------------------------------##
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_cfeat = 10, height = 28):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res = True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        self.up0 = nn.Sequential(
                                nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),
                                nn.GroupNorm(8, 2 * n_feat),
                                nn.ReLU(),
                                )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
                                nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
                                nn.GroupNorm(8, n_feat),
                                nn.ReLU(),
                                nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
                                )

    def forward(self, x, t, c = None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)

        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))

        return out

## ------------------------------------------------------##
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 64
n_cfeat = 5
height = 16
save_dir = './weights/'

batch_size = 100
n_epoch = 32
lrate = 1e-3

## ------------------------------------------------------##
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device = device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim = 0).exp()
ab_t[0] = 1

## ------------------------------------------------------##
nn_model = ContextUnet(in_channels = 3, n_feat = n_feat, n_cfeat = n_cfeat,
                       height = height).to(device)

## ------------------------------------------------------##
dataset = CustomDataset("./sprites_1788_16x16.npy", "./sprite_labels_nc_1788_16x16.npy", transform, null_context = False)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 1)
optim = torch.optim.Adam(nn_model.parameters(), lr = lrate)

## ------------------------------------------------------##
def perturb_input(x, t, noise):

    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

## ------------------------------------------------------##
def denoise_add_noise(x, t, pred_noise, z = None):
    if z is None:
        z = torch.randn_like(x)

    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()

    return mean + noise

## ------------------------------------------------------##
@torch.no_grad()
def sample_ddpm(n_sample, save_rate = 20):
    samples = torch.randn(n_sample, 3, height, height).to(device)

    intermediate = []

    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i : 3d}', end = '\r')

        t = torch.tensor([i / timesteps])[ : , None, None, None].to(device)

        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t)
        samples = denoise_add_noise(samples, i, eps, z)

        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)

    return samples, intermediate

## ------------------------------------------------------##
nn_model.load_state_dict(torch.load(f"{save_dir}/model_0.pth", map_location = device))
nn_model.eval()
print("Loaded in Model")

## ------------------------------------------------------##
plt.clf()
samples, intermediate_ddpm = sample_ddpm(32)
animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run", None, save = False)
HTML(animation_ddpm.to_jshtml())

## ------------------------------------------------------##
nn_model.load_state_dict(torch.load(f"{save_dir}/model_4.pth", map_location = device))
nn_model.eval()
print("Loaded in Model")

## ------------------------------------------------------##
plt.clf()
samples, intermediate_ddpm = sample_ddpm(32)
animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run", None, save = False)
HTML(animation_ddpm.to_jshtml())

## ------------------------------------------------------##
nn_model.load_state_dict(torch.load(f"{save_dir}/model_8.pth", map_location = device))
nn_model.eval()
print("Loaded in Model")

## ------------------------------------------------------##
plt.clf()
samples, intermediate_ddpm = sample_ddpm(32)
animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run", None, save = False)
HTML(animation_ddpm.to_jshtml())

## ------------------------------------------------------##
nn_model.load_state_dict(torch.load(f"{save_dir}/model_31.pth", map_location = device))
nn_model.eval()
print("Loaded in Model")

## ------------------------------------------------------##
plt.clf()
samples, intermediate_ddpm = sample_ddpm(32)
animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run", None, save = False)
HTML(animation_ddpm.to_jshtml())

## ------------------------------------------------------##
## Acknowledgments
# Sprites by ElvGames, [FrootsnVeggies](https://zrghr.itch.io/froots-and-veggies-culinary-pixels) and  [kyrise](https://kyrise.itch.io/)
# This code is modified from, https://github.com/cloneofsimo/minDiffusion
# Diffusion model is based on [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) and [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
