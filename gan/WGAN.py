import os

import torch
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    """

    loss_pt1 = torch.mean(discrim_fake) - torch.mean(discrim_real)
    gradients = torch.autograd.grad(outputs = discrim_interp,inputs =interp,grad_outputs = torch.ones_like(discrim_interp) ,retain_graph=True,create_graph=True)[0] 
    normalized_gradients = torch.norm(gradients.view(discrim_interp.shape[0],-1),dim =1)
    loss_pt2 = lamb *torch.mean((normalized_gradients-1)**2)
    loss = loss_pt1 + loss_pt2

    return loss


def compute_generator_loss(discrim_fake):

    loss = - torch.mean(discrim_fake)

    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
