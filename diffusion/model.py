import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)
from einops import rearrange, reduce

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.,
    ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas

        self.alphas_cumprod = torch.cumprod(alphas,dim =0 )
        self.alphas_cumprod_prev =   self.alphas_cumprod /alphas


        # This is the coefficient of x_t when predicting x_0
        self.x_0_pred_coef_1 = 1.0 / torch.sqrt(self.alphas_cumprod)
        # This is the coefficient of pred_noise when predicting x_0
        self.x_0_pred_coef_2 = - self.x_0_pred_coef_1 * torch.sqrt(1- self.alphas_cumprod)


        # This is coefficient of x_0 in the DDPM section
        self.posterior_mean_coef1 = (torch.sqrt(self.alphas_cumprod_prev)*self.betas)/(1- self.alphas_cumprod)
        # This is coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 = (torch.sqrt(alphas)*(1-self.alphas_cumprod_prev))/(1- self.alphas_cumprod)

  
        # Calculations for posterior q(x_{t-1} | x_t, x_0) in DDPM
        self.posterior_variance = (self.betas*(1-self.alphas_cumprod_prev))/(1- self.alphas_cumprod)

        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_posterior_parameters(self, x_0, x_t, t):
        # Compute the posterior mean and variance for x_{t-1}
        # using the coefficients, x_t, and x_0.
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_t, t):

        pred_noise = self.model(x_t,t)
        x_0 = x_t*extract(self.x_0_pred_coef_1,t,x_t.shape) +  \
        extract(self.x_0_pred_coef_2, t,pred_noise.shape)*pred_noise
        x_0 = torch.clamp(x_0,-1,1)

        return (pred_noise, x_0)

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):

        _, x_0 =  self.model_predictions(x,t)
        post_mean, post_var, post_log_clip_var =  self.get_posterior_parameters(x_0,x,t)
        z = torch.randn(post_mean.shape) if t.all() >0 else 0
        pred_img = post_mean + torch.sqrt(post_var)*z.cuda()

        return pred_img, x_0

    @torch.no_grad()
    def sample_ddpm(self, shape, z):
        img = z
        for t in tqdm(range(self.num_timesteps-1, 0, -1)):
            batched_times = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, batched_times)
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample_times(self, total_timesteps, sampling_timesteps):
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        return list(reversed(times.int().tolist()))

    def get_time_pairs(self, times):
        return list(zip(times[:-1], times[1:]))

    def ddim_step(self, batch, device, tau_i, tau_isub1, img, model_predictions, alphas_cumprod, eta):

        tau_i = torch.full((batch,),tau_i,device = self.device,dtype = torch.long)
        pred_noise, x_0 = model_predictions(img, tau_i)

        # Step 2: Extract \alpha_{\tau_{i - 1}} and \alpha_{\tau_{i}}
        if tau_isub1 >0:
            z =  torch.randn_like(img).cuda()
        else:
            return img,x_0

        tau_isub1 =  torch.full((batch,),tau_isub1,device = self.device,dtype = torch.long)
        alpha_i = extract(alphas_cumprod, tau_i, x_0.shape)
        alpha_isub1 =  extract(alphas_cumprod, tau_isub1, x_0.shape)
        # Step 3: Compute \sigma_{\tau_{i}}
        sigma_i = eta*(1-alpha_isub1) *(extract(self.betas, tau_isub1,x_0.shape))/(1-alpha_i)

        # Step 4: Compute the coefficient of \epsilon_{\tau_{i}}
        coeff = torch.sqrt(1-alpha_isub1- sigma_i)

        # Step 5: Sample from q(x_{\tau_{i - 1}} | x_{\tau_t}, x_0)
        # HINT: Use the reparameterization trick
        Mean_i =  torch.sqrt(alpha_isub1)*x_0 + pred_noise* coeff
        img = Mean_i + torch.sqrt(sigma_i)*z   


        return img, x_0

    def sample_ddim(self, shape, z):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = self.sample_times(total_timesteps, sampling_timesteps)
        time_pairs = self.get_time_pairs(times)

        img = z
        for tau_i, tau_isub1 in tqdm(time_pairs, desc='sampling loop time step'):
            img, _ = self.ddim_step(batch, device, tau_i, tau_isub1, img, self.model_predictions, self.alphas_cumprod, eta)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = torch.randn(shape, device = self.betas.device)
        return sample_fn(shape, z)

    @torch.no_grad()
    def sample_given_z(self, z, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = z.reshape(shape)
        return sample_fn(shape, z)
