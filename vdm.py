import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, rand, randn_like, sigmoid, sqrt, tensor

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
        """
        Encode discrete image values to continuous latent space.
        Transforms from [0, 1] to approximately [-1, 1].

        x: (B, C, H, W) in [0, 1] range (from ToTensor())
        returns: (B, C, H, W) in approximately [-1, 1]
        """
        # Round to ensure discrete values, then rescale
        # ToTensor gives [0, 1], we map to [-1, 1]
        x_discrete = (x * 255).round()
        return 2 * ((x_discrete + 0.5) / self.vocab_size) - 1

    def decode(self, z: Tensor, g_0: Tensor) -> Tensor:
        """
        Compute log probabilities for reconstruction.

        z: latent representation (B, C, H, W)
        g_0: gamma at t=0, scalar or (B,)
        returns: log probabilities (B, C, H, W, vocab_size)
        """
        # Create all possible discrete values
        x_vals = torch.arange(0, self.vocab_size, device=z.device, dtype=z.dtype)  # (vocab_size,)
        x_vals = x_vals.view(1, 1, 1, 1, self.vocab_size)  # (1, 1, 1, 1, vocab_size)

        # Encode them to continuous space
        x_vals_encoded = 2 * ((x_vals + 0.5) / self.vocab_size) - 1  # (1, 1, 1, 1, vocab_size)

        # Compute log probabilities using Gaussian likelihood
        # Expand g_0 to match z dimensions
        if g_0.dim() == 0:
            g_0 = g_0.expand(z.shape[0])
        inv_stdev = torch.exp(-0.5 * g_0)  # (B,)
        inv_stdev = inv_stdev.view(-1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1)

        # z: (B, C, H, W) -> (B, C, H, W, 1)
        z_expanded = z.unsqueeze(-1)

        # Compute negative squared distance
        logits = -0.5 * torch.square((z_expanded - x_vals_encoded) * inv_stdev)  # (B, C, H, W, vocab_size)

        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (B, C, H, W, vocab_size)
        return log_probs

    def compute_reconstruction_loss(self, x: Tensor, z_0_rescaled: Tensor, g_0: Tensor) -> Tensor:
        """
        Compute the reconstruction loss: -log p(x | z_0).

        x: original images (B, C, H, W) in [0, 1]
        z_0_rescaled: rescaled latent at t=0 (B, C, H, W)
        g_0: gamma at t=0
        returns: loss per sample (B,)
        """
        # Get discrete values
        x_discrete = (x * 255).round().long()  # (B, C, H, W)

        # Get log probabilities
        log_probs = self.decode(z_0_rescaled, g_0)  # (B, C, H, W, vocab_size)

        # Convert to one-hot and compute log probability
        x_onehot = F.one_hot(x_discrete, num_classes=self.vocab_size).float()  # (B, C, H, W, vocab_size)

        # Sum log probabilities across all dimensions
        log_prob = torch.sum(x_onehot * log_probs, dim=[1, 2, 3, 4])  # (B,)

        # Return negative log likelihood
        return -log_prob

    def compute_kl_loss(self, f: Tensor, var_1: float) -> Tensor:
        """
        Compute KL divergence between q(z_1|x_0) and prior p(z_1) = N(0, I).

        f: encoded data (B, C, H, W)
        var_1: variance at t=1
        returns: KL divergence per sample (B,)
        """
        # KL(q(z_1|x_0) || N(0, I))
        # q(z_1|x_0) = N(sqrt(1-var_1) * f, var_1 * I)
        mean1_sqr = (1.0 - var_1) * torch.square(f)
        var_1_tensor = torch.tensor(var_1, device=f.device, dtype=f.dtype)
        kl = 0.5 * torch.sum(mean1_sqr + var_1_tensor - torch.log(var_1_tensor) - 1.0, dim=[1, 2, 3])
        return kl

    def compute_diffusion_loss(self, eps: Tensor, eps_pred: Tensor, g_t: Tensor, t: Tensor) -> Tensor:
        """
        Compute the diffusion loss with proper VDM weighting.

        eps: true noise (B, C, H, W)
        eps_pred: predicted noise (B, C, H, W)
        g_t: gamma at time t (B,)
        t: timesteps (B,)
        returns: weighted diffusion loss per sample (B,)
        """
        # Compute MSE per sample
        loss_diff_mse = torch.sum(torch.square(eps - eps_pred), dim=[1, 2, 3])  # (B,)

        # For continuous time (T=0), weight by derivative of gamma
        # We need to compute d(gamma)/dt
        # Since gamma is linear: gamma(t) = gamma_min + (gamma_max - gamma_min) * t
        # d(gamma)/dt = gamma_max - gamma_min
        g_t_grad = GAMMA_MAX - GAMMA_MIN

        # Weight the loss
        loss_diff = 0.5 * g_t_grad * loss_diff_mse  # (B,)

        return loss_diff

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
        times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
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
        Compute the full VDM ELBO with all three components.

        batch: (B, C, H, W) in [0, 1] range (from ToTensor())
        noise: optional noise tensor for reproducibility

        Returns:
            total_loss: scalar loss for backprop (mean across batch)
            bpd: total BPD (ELBO) per sample (mean across batch)
            bpd_components: dict with individual BPD components
        """
        x = batch
        batch_size = x.shape[0]

        # Get gamma values at boundaries
        g_0 = self.gamma(torch.tensor(0.0, device=x.device))
        g_1 = self.gamma(torch.tensor(1.0, device=x.device))
        var_0 = sigmoid(g_0)
        var_1 = sigmoid(g_1)

        # 1. ENCODE: Transform discrete data to continuous latent
        f = self.encode(x)  # (B, C, H, W)

        # 2. RECONSTRUCTION LOSS
        # Sample z_0 ~ q(z_0 | x)
        if noise is None:
            eps_0 = torch.randn_like(f)
        else:
            eps_0 = noise
        z_0 = sqrt(1.0 - var_0) * f + sqrt(var_0) * eps_0
        z_0_rescaled = f + torch.exp(0.5 * g_0) * eps_0  # = z_0 / sqrt(1 - var_0)

        loss_recon = self.compute_reconstruction_loss(x, z_0_rescaled, g_0)  # (B,)

        # 3. KL LOSS (latent prior)
        loss_klz = self.compute_kl_loss(f, var_1.item())  # (B,)

        # 4. DIFFUSION LOSS
        # Sample timesteps
        t = self.sample_times(batch_size)

        # Sample z_t ~ q(z_t | x)
        z_t, eps, gamma_t = self.q_sample(f, t, noise=None)

        # Predict noise
        eps_pred = self.model(z_t, gamma_t)

        # Compute weighted diffusion loss
        loss_diff = self.compute_diffusion_loss(eps, eps_pred, gamma_t, t)  # (B,)

        # 5. COMPUTE TOTAL LOSS AND BPD
        # Total loss in nats (sum of three components)
        total_loss_per_sample = loss_recon + loss_klz + loss_diff  # (B,)

        # Convert to bits per dimension
        num_dims = np.prod(self.image_shape)
        rescale_to_bpd = 1.0 / (num_dims * np.log(2.0))

        bpd_recon = torch.mean(loss_recon) * rescale_to_bpd
        bpd_klz = torch.mean(loss_klz) * rescale_to_bpd
        bpd_diff = torch.mean(loss_diff) * rescale_to_bpd
        bpd_total = bpd_recon + bpd_klz + bpd_diff

        # For backprop, use mean of total loss
        total_loss = torch.mean(total_loss_per_sample)

        # Return components for logging
        bpd_components = {
            "bpd_recon": bpd_recon.item(),
            "bpd_klz": bpd_klz.item(),
            "bpd_diff": bpd_diff.item(),
        }

        return total_loss, bpd_total, bpd_components


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
