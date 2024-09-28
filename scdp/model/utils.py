import torch 
import torch.optim as optim
from bisect import bisect

from scdp.common.utils import scatter

def get_nmape(pred_density, density, batch=None):
    diff = pred_density - density
    sum_over = list(range(1, diff.dim()))
    if batch is None:
        return torch.abs(diff).sum(dim=sum_over) / torch.abs(density).sum(dim=sum_over) * 100.
    else:
        return (scatter(diff.abs(), batch, dim_size=batch.max() + 1).abs() / 
                scatter(density.abs(), batch, dim_size=batch.max() + 1).abs() * 100.)

def warmup_lr_milestone(current_step: int, optim_config):
    """
    adapted from https://github.com/Open-Catalyst-Project/ocp
    Returns a learning rate multiplier.
    Till `warmup_steps`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by `lr_gamma` every time a milestone is crossed.
    """

    # keep this block for older configs that have warmup_epochs instead of warmup_steps
    # and lr_milestones are defined in epochs
    if (
        any(x < 100 for x in optim_config["lr_milestones"])
        or "warmup_epochs" in optim_config
    ):
        raise Exception(
            "ConfigError: please define lr_milestones in steps not epochs and define warmup_steps instead of warmup_epochs"
        )

    if current_step <= optim_config["warmup_steps"]:
        alpha = current_step / float(optim_config["warmup_steps"])
        return optim_config["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(optim_config["lr_milestones"], current_step)
        return pow(optim_config["lr_gamma"], idx)
    
def warm_lr_lambda(current_step: int, 
                   warmup_steps: int = 0, 
                   warmup_factor: float = 0.2,
                   alpha: float = 0.96, 
                   beta: float = 1e4):
    if current_step <= warmup_steps:
        alpha = current_step / float(warmup_steps)
        return warmup_factor * (1.0 - alpha) + alpha
    else:
        return alpha ** (current_step / beta)
            
class PowerDecayScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps=1000, warmup_factor=0.2, alpha=0.96, beta=1e4):
        scheduler_fn = lambda step: warm_lr_lambda(step, warmup_steps, warmup_factor, alpha, beta)
        super().__init__(optimizer=optimizer, lr_lambda=scheduler_fn)

class WarmupMilestoneScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, config):
        self.config = config
        scheduler_fn = lambda step: warmup_lr_milestone(step, self.config)
        super().__init__(optimizer=optimizer, lr_lambda=scheduler_fn)
    