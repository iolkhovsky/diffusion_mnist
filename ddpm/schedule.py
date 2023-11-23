from dataclasses import dataclass, fields
import torch

@dataclass
class Schedule:
    beta1: float
    beta2: float
    steps: int
    alpha_t: torch.Tensor
    oneover_sqrta: torch.Tensor
    sqrt_beta_t: torch.Tensor
    alphabar_t: torch.Tensor
    sqrtab: torch.Tensor
    sqrtmab: torch.Tensor
    mab_over_sqrtmab: torch.Tensor

    @classmethod
    def from_betas(cls, beta1, beta2, steps):
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = (beta2 - beta1) * torch.arange(0, steps + 1, dtype=torch.float32) / steps + beta1
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return Schedule(
            beta1=beta1,
            beta2=beta2,
            steps=steps,
            alpha_t=alpha_t,
            oneover_sqrta=oneover_sqrta,
            sqrt_beta_t=sqrt_beta_t,
            alphabar_t=alphabar_t,
            sqrtab=sqrtab,
            sqrtmab=sqrtmab,
            mab_over_sqrtmab=mab_over_sqrtmab_inv,
        )

    def to(self, device):
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)
            if isinstance(field_value, torch.Tensor):
                setattr(self, field_name, field_value.to(device))
        return self
