import torch

class Diffusion:
    def __init__(self, time_steps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.betas = torch.linspace(beta_start, beta_end, time_steps).float().to(device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.one_minus_sqrt_alphas_cumprod = 1. - torch.sqrt(self.alphas_cumprod)

    @staticmethod
    def _extract(data, batch_t, shape):
        batch_size = batch_t.shape[0]
        out = torch.gather(data, -1, batch_t)

        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start, trend, batch_t, noise):
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, batch_t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, batch_t, x_start.shape)
        one_minus_sqrt_alphas_cumprod_t = self._extract(self.one_minus_sqrt_alphas_cumprod, batch_t, x_start.shape)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise + one_minus_sqrt_alphas_cumprod_t * trend
        # x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy


    def q_sample_directionalNoise(self, x_start, trend, batch_t, noise):
        x_start_mean = torch.mean(x_start)
        x_start_sigma = torch.var(x_start)
        directionalNOise_1 = x_start_mean + x_start_sigma * noise
        directionalNOise_2 = torch.sign(x_start_mean - x_start) * torch.abs(directionalNOise_1)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, batch_t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, batch_t, x_start.shape)
        one_minus_sqrt_alphas_cumprod_t = self._extract(self.one_minus_sqrt_alphas_cumprod, batch_t, x_start.shape)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * directionalNOise_2 + one_minus_sqrt_alphas_cumprod_t * trend
        return x_noisy
