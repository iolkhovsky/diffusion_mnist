import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from ddpm.unet import ContextUnet
from ddpm.schedule import Schedule


class DDPM(pl.LightningModule):
    def __init__(self, in_channels=1, hidden_channels=128, n_classes=10,
                 beta1=1e-4, beta2=2e-2, steps=400, drop_prob=0.1):
        super().__init__()
        self.core = ContextUnet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_classes=n_classes,
        )
        self.schedule = Schedule.from_betas(
            beta1=beta1,
            beta2=beta2,
            steps=steps,
        ).to(self.device)
        self.drop_prob = drop_prob
        self.criterion = nn.MSELoss()

    def add_noise(self, x: torch.Tensor, timestamp: torch.LongTensor):
        timestamp = timestamp.to(self.device)
        noise = torch.randn_like(x).to(self.device)
        self.schedule.to(self.device)
        x_t = (
            self.schedule.sqrtab[timestamp, None, None, None] * x
            + self.schedule.sqrtmab[timestamp, None, None, None] * noise
        )
        return x_t, noise

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x: torch.Tensor, c: torch.LongTensor, t: torch.LongTensor, mask: torch.LongTensor):
        return self.core(x, c, t / self.schedule.steps, mask)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, contexts = batch
        batch_size = images.shape[0]
        timestamps = torch.randint(
            1, self.schedule.steps + 1, (batch_size,)
        ).to(self.device)

        noisy_images, noise_fields = self.add_noise(images, timestamps)
        context_mask = torch.bernoulli(
            torch.zeros_like(contexts) + self.drop_prob
        ).to(self.device)

        predicted_noise = self.forward(noisy_images, contexts, timestamps, context_mask)
        loss = self.criterion(predicted_noise, noise_fields)
        self.log('loss/mse', loss)

        return loss

    def sample(self, n_sample, size, device, guide_w=0.0):
        x_i = torch.randn(n_sample, *size).to(device)
        c_i = torch.arange(0, 10).to(device)
        c_i = c_i.repeat(int(n_sample / c_i.shape[0]))

        context_mask = torch.zeros_like(c_i).to(device)

        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.

        with tqdm(total=self.schedule.steps) as pbar:
            for step_idx in range(self.schedule.steps):
                pbar.set_description(f'Diffusion step {step_idx}')
                i = self.schedule.steps - 1 - step_idx

                t_is = torch.tensor([i / self.schedule.steps]).to(device)
                t_is = t_is.repeat(n_sample,1,1,1)

                x_i = x_i.repeat(2,1,1,1)
                t_is = t_is.repeat(2,1,1,1)

                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

                eps = self.core(x_i, c_i, t_is.view(-1), context_mask)
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + guide_w) * eps1 - guide_w * eps2
                x_i = x_i[:n_sample]
                x_i = (
                    self.schedule.oneover_sqrta[i] * (x_i - eps * self.schedule.mab_over_sqrtmab[i])
                    + self.schedule.sqrt_beta_t[i] * z
                )
                pbar.update(1)
        
        return x_i

    def validation_step(self, batch, batch_idx: int) -> None:
        writer = self.logger.experiment
        ws_test = [0.0, 0.5, 2.0]
        n_classes = 10
        n_sample = 4 * n_classes
        for w_i, w in enumerate(ws_test):
            x_gen = self.sample(n_sample, (1, 28, 28), self.device, guide_w=w)
            genereted_samples = make_grid(x_gen*-1 + 1, nrow=10)
            writer.add_image(f'Samples(w={w_i})', genereted_samples, self.global_step)


            # save_image(grid, self.logger.log_dir + f"image_ep{self.global_step}_w{w}.png")
