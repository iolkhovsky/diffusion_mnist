import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from tqdm import tqdm

from ddpm.unet import ContextUnet
from ddpm.schedule import Schedule
from utils import images_to_grid, save_gif


class DdpmPreprocessor(nn.Module):
    def __init__(self, scale: float, channels: int):
        super().__init__()
        self.scale = scale
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4 and x.shape[1] == self.channels
        return x.float() * self.scale


class DdpmPostprocessor(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x / self.scale, 0., 255.).to(torch.uint8)


class DDPM(pl.LightningModule):
    def __init__(self, in_channels=1, hidden_channels=128, n_classes=10,
                 beta1=1e-4, beta2=2e-2, steps=400, drop_prob=0.1, scale=1./255.,
                 lr=1e-4):
        super().__init__()
        self.core = ContextUnet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_classes=n_classes,
        )
        self.scale = scale
        self.lr = lr
        self.preprocessor = DdpmPreprocessor(self.scale, in_channels)
        self.postprocessor = DdpmPostprocessor(self.scale)
        self.schedule = Schedule.from_betas(
            beta1=beta1,
            beta2=beta2,
            steps=steps,
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.drop_prob = drop_prob

    def decay_image(self, x: torch.Tensor, timestamp: torch.LongTensor):
        batch_size = x.shape[0]
        assert batch_size == timestamp.shape[0]

        timestamp = timestamp.to(self.device)
        noise = torch.randn_like(x).to(self.device)
        self.schedule.to(self.device)

        x_t = (
            self.schedule.sqrtab[timestamp, None, None, None] * x
            + self.schedule.sqrtmab[timestamp, None, None, None] * noise
        )
        return x_t, noise

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, contexts = batch
        images = self.preprocessor(images)

        batch_size = images.shape[0]
        timestamps = torch.randint(
            1, self.schedule.steps + 1, (batch_size,)
        ).to(self.device)
        noisy_images, noise_fields = self.decay_image(images, timestamps)

        mask_context = torch.bernoulli(
            torch.zeros_like(contexts) + self.drop_prob
        ).to(self.device)

        predicted_noise = self.core(noisy_images, contexts, timestamps / self.schedule.steps, mask_context)

        loss = self.criterion(predicted_noise, noise_fields)
        self.log('loss/mse', loss)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, contexts: torch.LongTensor,
                guide_w: torch.Tensor,
                return_history: bool = False) -> torch.Tensor:
        assert len(contexts.shape) == 1
        batch_size = contexts.shape[0]

        images = torch.randn(batch_size, 1, 28, 28).to(self.device)
        contexts = contexts.to(self.device)
        contexts_mask = torch.cat(
            [
                torch.zeros_like(contexts).float(),
                torch.ones_like(contexts).float(),
            ]
        ).to(self.device)
        contexts = contexts.repeat(2)

        history = []
        if return_history:
            history.append(self.postprocessor(images).detach().cpu().numpy())

        with tqdm(total=self.schedule.steps) as pbar:
            for timestamp_idx in range(self.schedule.steps - 1, 0, -1):
                pbar.set_description(f'Diffusion step {timestamp_idx}')

                norm_tstamp = torch.tensor([timestamp_idx / self.schedule.steps]).to(self.device)
                norm_tstamp = norm_tstamp.repeat(batch_size, 1, 1, 1)

                images = images.repeat(2, 1, 1, 1)
                i_tstamps = norm_tstamp.repeat(2, 1, 1, 1)
                z = torch.randn(batch_size, 1, 28, 28).to(self.device) if timestamp_idx > 1 else 0

                pred_noise = self.core(images, contexts, i_tstamps.view(-1), contexts_mask)
                eps1, eps2 = pred_noise[:batch_size], pred_noise[batch_size:]
                noise = (1 + guide_w) * eps1 - guide_w * eps2

                images = images[:batch_size]
                img_k = self.schedule.oneover_sqrta[timestamp_idx]
                noise_k = self.schedule.sqrt_beta_t[timestamp_idx]

                images = img_k * (
                    images - noise * self.schedule.mab_over_sqrtmab[timestamp_idx]
                ) + noise_k * z

                if return_history:
                    if timestamp_idx % 20 == 0 or timestamp_idx < 8:
                        history.append(self.postprocessor(images).detach().cpu().numpy())

                pbar.update(1)

        out_images = self.postprocessor(images)

        if return_history:
            return out_images, history
        else:
            return out_images

    def validation_step(self, batch, batch_idx: int) -> None:
        writer = self.logger.experiment
        for guide_w in [0., 0.5, 2.]:
            guide_w = torch.tensor(guide_w).to(self.device)
            contexts = torch.tensor(
                list(range(10)) * 4
            ).long().to(self.device)
            images, history = self.forward(contexts, guide_w, True)
            genereted_samples = make_grid(images, nrow=10)
            writer.add_image(f'Samples(w={guide_w})', genereted_samples, self.global_step)
            history = [
                images_to_grid(np.transpose(x, axes=(0, 2, 3, 1)))
                for x in history
            ]  # b, c, h, w -> b, h, w, c
            save_gif(history, f'validation_guide_{guide_w}_step_{self.global_step}.gif')
