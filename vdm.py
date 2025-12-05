import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, randn_like, sigmoid, sqrt, tensor
from torch.special import expm1
from tqdm import trange

GAMMA_MIN = -13.3
GAMMA_MAX = 5.0


class VDM(nn.Module):
    def __init__(self, model, image_shape, device=None):
        super().__init__()
        self.model = model
        self.image_shape = image_shape
        self.vocab_size = 256
        self.gamma = FixedLinearSchedule(GAMMA_MIN, GAMMA_MAX)
        self.device = device

    def encode(self, x: Tensor) -> Tensor:
        # [0, 1] -> [-1. 1]
        x_discrete = (x * 255).round()
        return 2 * ((x_discrete + 0.5) / self.vocab_size) - 1

    def decode(self, z: Tensor, g_0: Tensor) -> Tensor:

        x_vals = torch.arange(0, self.vocab_size, device=z.device, dtype=z.dtype)
        x_vals = x_vals.view(1, 1, 1, 1, self.vocab_size)

        x_vals_encoded = 2 * ((x_vals + 0.5) / self.vocab_size) - 1

        if g_0.dim() == 0:
            g_0 = g_0.expand(z.shape[0])
        inv_stdev = torch.exp(-0.5 * g_0)
        inv_stdev = inv_stdev.view(-1, 1, 1, 1, 1)

        z_expanded = z.unsqueeze(-1)

        logits = -0.5 * torch.square((z_expanded - x_vals_encoded) * inv_stdev)

        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def compute_reconstruction_loss(self, x: Tensor, z_0_rescaled: Tensor, g_0: Tensor) -> Tensor:

        x_discrete = (x * 255).round().long()

        log_probs = self.decode(z_0_rescaled, g_0)

        x_onehot = F.one_hot(x_discrete, num_classes=self.vocab_size).float()

        log_prob = torch.sum(x_onehot * log_probs, dim=[1, 2, 3, 4])

        return -log_prob

    def compute_kl_loss(self, f: Tensor, var_1: float) -> Tensor:
        mean1_sqr = (1.0 - var_1) * torch.square(f)
        var_1_tensor = torch.tensor(var_1, device=f.device, dtype=f.dtype)
        kl = 0.5 * torch.sum(mean1_sqr + var_1_tensor - torch.log(var_1_tensor) - 1.0, dim=[1, 2, 3])
        return kl

    def compute_diffusion_loss(self, eps: Tensor, eps_pred: Tensor) -> Tensor:

        loss_diff_mse = torch.sum(torch.square(eps - eps_pred), dim=[1, 2, 3])

        g_t_grad = GAMMA_MAX - GAMMA_MIN

        loss_diff = 0.5 * g_t_grad * loss_diff_mse

        return loss_diff

    def alpha_sigma(self, t: Tensor):
        """
        Given t in [0,1], return alpha_t, sigma_t (both shape like t).
        """

        # Taken from VDM codebase
        g_t = self.gamma(t)
        alpha2 = sigmoid(-g_t)
        sigma2 = sigmoid(g_t)
        return sqrt(alpha2), sqrt(sigma2)

    def sample_times(self, batch_size):
        t0 = np.random.uniform(0, 1 / batch_size)
        times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
        return times

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None):
        with torch.enable_grad():
            gamma_t = self.gamma(t)

        def unsqueeze_right(x: Tensor, num_dims=1):
            return x.view(x.shape + (1,) * num_dims)

        gamma_t_padded = unsqueeze_right(gamma_t, x0.ndim - gamma_t.ndim)

        mean = x0 * sqrt(sigmoid(-gamma_t_padded))
        scale = sqrt(sigmoid(gamma_t_padded))

        if noise is None:
            noise = randn_like(x0)

        z_t = mean + noise * scale
        return z_t, noise, gamma_t

    def forward(self, batch, *, noise=None):
        x = batch
        batch_size = x.shape[0]

        # Get gamma values at boundaries
        g_0 = self.gamma(torch.tensor(0.0, device=x.device))
        g_1 = self.gamma(torch.tensor(1.0, device=x.device))
        var_0 = sigmoid(g_0)
        var_1 = sigmoid(g_1)

        f = self.encode(x)

        if noise is None:
            eps_0 = torch.randn_like(f)
        else:
            eps_0 = noise
        z_0 = sqrt(1.0 - var_0) * f + sqrt(var_0) * eps_0
        z_0_rescaled = z_0 / sqrt(1.0 - var_0)

        loss_recon = self.compute_reconstruction_loss(x, z_0_rescaled, g_0)

        loss_klz = self.compute_kl_loss(f, var_1.item())

        t = self.sample_times(batch_size)

        z_t, eps, gamma_t = self.q_sample(f, t, noise=None)

        # Predict noise
        eps_pred = self.model(z_t, gamma_t)

        loss_diff = self.compute_diffusion_loss(eps, eps_pred)

        total_loss_per_sample = loss_recon + loss_klz + loss_diff

        num_dims = np.prod(self.image_shape)
        rescale_to_bpd = 1.0 / (num_dims * np.log(2.0))

        bpd_recon = torch.mean(loss_recon) * rescale_to_bpd
        bpd_klz = torch.mean(loss_klz) * rescale_to_bpd
        bpd_diff = torch.mean(loss_diff) * rescale_to_bpd
        bpd_total = bpd_recon + bpd_klz + bpd_diff

        # For backprop, use mean of total loss
        total_loss = torch.mean(total_loss_per_sample)

        # For logging
        bpd_components = {
            "bpd_recon": bpd_recon.item(),
            "bpd_klz": bpd_klz.item(),
            "bpd_diff": bpd_diff.item(),
        }

        return total_loss, bpd_total, bpd_components

    @torch.no_grad()
    def sample_p_s_t(self, z, t, s, clip_samples):
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        alpha_t = sqrt(sigmoid(-gamma_t))
        alpha_s = sqrt(sigmoid(-gamma_s))
        sigma_t = sqrt(sigmoid(gamma_t))
        sigma_s = sqrt(sigmoid(gamma_s))

        pred_noise = self.model(z, gamma_t)
        if clip_samples:
            x_start = (z - sigma_t * pred_noise) / alpha_t
            x_start.clamp_(-1.0, 1.0)
            mean = alpha_s * (z * (1 - c) / alpha_t + c * x_start)
        else:
            mean = alpha_s / alpha_t * (z - c * sigma_t * pred_noise)
        scale = sigma_s * sqrt(c)
        return mean + scale * torch.randn_like(z)

    def log_probs_x_z0(self, x=None, z_0=None):
        # Adapted from addtt/variational-diffusion-models

        gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        if x is None and z_0 is not None:
            z_0_rescaled = z_0 / sqrt(sigmoid(-gamma_0))

        elif z_0 is None and x is not None:
            z_0_rescaled = x + torch.exp(0.5 * gamma_0) * torch.randn_like(x)

        else:
            raise ValueError("Must provide either x or z_0, not both.")

        z_0_rescaled = z_0_rescaled.unsqueeze(-1)
        x_lim = 1 - 1 / self.vocab_size
        x_values = torch.linspace(-x_lim, x_lim, self.vocab_size, device=self.device)
        logits = -0.5 * torch.exp(-gamma_0) * (z_0_rescaled - x_values) ** 2
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs

    @torch.no_grad()
    def sample(self, batch_size, n_sample_steps=250, clip_samples=True):
        # Adapted from addtt/variational-diffusion-models
        z = torch.randn((batch_size, *self.image_shape), device=self.device)
        steps = torch.linspace(1.0, 0.0, n_sample_steps + 1, device=self.device)
        for i in trange(n_sample_steps, desc="sampling"):
            z = self.sample_p_s_t(z, steps[i], steps[i + 1], clip_samples)
        logprobs = self.log_probs_x_z0(z_0=z)
        x = torch.argmax(logprobs, dim=-1)
        return x.float() / (self.vocab_size - 1)

    @torch.no_grad()
    def reconstruct(self, real_batch: Tensor, t_start: float, n_steps: int) -> Tensor:
        f = self.encode(real_batch)

        batch_size = f.shape[0]
        t = torch.full((batch_size,), t_start, device=self.device)
        z_t, _, _ = self.q_sample(f, t, noise=None)

        steps = torch.linspace(t_start, 0.0, n_steps + 1, device=self.device)
        z = z_t
        for i in trange(n_steps, desc="reconstruction"):
            z = self.sample_p_s_t(z, steps[i], steps[i + 1], clip_samples=True)

        log_probs = self.log_probs_x_z0(z_0=z)
        x_indices = torch.argmax(log_probs, dim=-1)
        reconstructed = x_indices.float() / (self.vocab_size - 1)

        return reconstructed


class FixedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t
