import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision

def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):

    
    z = torch.zeros(100,128).cuda()
    gridx,gridy = torch.meshgrid(torch.linspace(-1,1,10),torch.linspace(-1,1,10))
    z[:,0] = gridx.reshape(-1).cuda()
    z[:,1] = gridy.reshape(-1).cuda()
    
    out =  gen.forward_given_samples(z)
    out = (out + 1 )/2
    torchvision.utils.save_image(out, path,nrow=10)


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--disable_amp", action="store_true", default=True)
    args = parser.parse_args()
    return args
