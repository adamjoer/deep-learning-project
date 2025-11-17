import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, rand, randn_like, sigmoid, sqrt, tensor

GAMMA_MIN = -13.3
GAMMA_MAX = 5.0


class VDM(nn.Module):
    def __init__(self, model, image_shape):
        super().__init__()
        self.model = model
        self.image_shape = image_shape
        self.vocab_size = 256
        self.gamma = FixedLinearSchedule(GAMMA_MIN, GAMMA_MAX)

    def alpha_sigma(self, t: Tensor):
        """
        Given t in [0,1], return alpha_t, sigma_t (both shape like t).
        """
        g_t = self.gamma(t)  # (B,)
        alpha2 = sigmoid(-g_t)
        sigma2 = sigmoid(g_t)
        return sqrt(alpha2), sqrt(sigma2)

    def sample_times(self, batch_size):
        t0 = np.random.uniform(0, 1 / batch_size)
        times = np.arange(t0, 1.0, 1.0 / batch_size)
        return times

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None):
        """
        Sample z_t ~ q(z_t | x0, t) = N(alpha_t * x0, sigma_t^2 * I).

        x0: (B, C, H, W), assumed in [-1, 1]
        t:  (B,) in [0, 1]
        noise: optional noise tensor, (B, C, H, W)
        """
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            gamma_t = self.gamma(t)

        # Assert that gamma_t has shape (B, 1, 1, 1) for broadcasting

        def unsqueeze_right(x, num_dims=1):
            """Unsqueezes the last `num_dims` dimensions of `x`."""
            return x.view(x.shape + (1,) * num_dims)

        gamma_t_padded = unsqueeze_right(gamma_t, x0.ndim - gamma_t.ndim)

        mean = x0 * sqrt(sigmoid(-gamma_t_padded))  # x * alpha_t
        scale = sqrt(sigmoid(gamma_t_padded))  # sigma_t

        if noise is None:
            noise = randn_like(x0)

        z_t = mean + noise * scale
        return z_t, noise, gamma_t

    def forward(self, batch, *, noise=None):
        """
        Given a batch of data x0, compute the VDM loss.

        batch: (B, C, H, W), assumed in [-1, 1]
        noise: optional noise tensor, (B, C, H, W)
        """

        # def maybe_unpack_batch(batch):
        #     if isinstance(batch, (tuple, list)) and len(batch) == 2:
        #         return batch
        #     else:
        #         return batch, None

        # x, label = maybe_unpack_batch(batch)

        # 1. sample times
        t = self.sample_times(batch.shape[0])

        # 2. forward noising
        z_t, eps, gamma_t = self.q_sample(batch, t, noise=noise)

        # UNet.forward expects (z, g_t) where g_t = gamma(t)

        # 3. predict noise with model
        eps_pred = self.model(z_t, gamma_t)

        # 4. diffusion loss (simple version, VDM-style weighting can come later)
        loss = F.mse_loss(eps_pred, eps, reduction="mean")
        return loss


class FixedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t


class LearnedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.b = nn.Parameter(tensor(gamma_min))
        self.w = nn.Parameter(tensor(gamma_max - gamma_min))

    def forward(self, t):
        return self.b + self.w.abs() * t
