# Implementations of WGAN-GP, LSGAN, VANILLA GAN, WGAN 

*1.1*. Let's start by setting up our networks for training a Generative Adversarial Network (GAN).  GANs have two networks, a generator and a discriminator. The generator takes in a noise sample z, generally sampled from the standard normal distribution, and maps it to an image. The discriminator takes in images and outputs the probability that the image is real or fake.

*1.2*. In general, we train the generator such that it can fool the discriminator, i.e. samples from the generator will have high probability under the discriminator. Analogously, we train the discriminator such that it can tell apart real and fake images. This means our loss term encourages the discriminator to assign high probability to real images while assigning low probability to fake images. T


## Relevant papers:
[1] Generative Adversarial Nets (Goodfellow et al, 2014): https://arxiv.org/pdf/1406.2661.pdf

[2] Least Squares Generative Adversarial Networks (Mao etclassification al, 2016): https://arxiv.org/pdf/1611.04076.pdf

[3] Improved Training of Wasserstein GANs (Gulrajani et al, 2017): https://arxiv.org/pdf/1704.00028.pdf

