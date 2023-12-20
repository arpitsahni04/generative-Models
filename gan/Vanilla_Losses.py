import os
# os.environ['PYTORCH_JIT'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import os
from utils import get_args

import torch
from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):

    real_targets = torch.ones_like(discrim_real)
    fake_targets = torch.zeros_like(discrim_fake)
    real_loss =  F.binary_cross_entropy_with_logits(discrim_real,real_targets)
    fake_loss = F.binary_cross_entropy_with_logits(discrim_fake,fake_targets)

    loss = real_loss+ fake_loss

    return loss


def compute_generator_loss(discrim_fake):

    loss = F.binary_cross_entropy_with_logits(discrim_fake,torch.ones_like(discrim_fake).cuda())

    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
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
