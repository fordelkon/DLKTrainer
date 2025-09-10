from dataclasses import dataclass, field
from typing import List


@dataclass
class BaseTrainerConfig:
    device: int = 0 # device to use, e.g. 0 or 1 or 2 for single GPU, if not GPU, switch to CPU

    ##### optimizer & scheduler #####
    optimizer: str = "Adam" # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp]
    scheduler: str = "Cosine" # learning rate scheduler to use, choices=[Cosine, Linear, Exponential, Polynomial]
    initial_lr: float = 3e-4 # initial learning rate
    f2i_lr: float = 0.1 # final_lr = initial_lr * f2i_lr
    gamma: float = 0.9 # < 1.0 the decay factor each epoch learning rate (lr0 * gamma**epochs) used in Exponential
    power: float = 2.0 # controls the decay curve shape used in Polynomial
    milestones: List[int] = field(default_factory=lambda: [50, 100, 150]) # epoch indices at which the learning rate will be decayed, used in MultiStep schedule.  
    step_ratio: float = 0.1 # multiplicative factor of learning rate decay at each milestone or step used in MultiStep scheduler
    n_cycles: int = 1 # number of cycles to run for multi-cyclic 
    cycle_scheduler: List[str] = field(default_factory=lambda: ["Cosine"]) # type(s) of scheduler function used within each cycle (e.g., "Cosine", "Linear", "Constant", "Polynominal", "Exponential"). 
    momentum: float = 0.937 # SGD momentum/Adam beta1
    weight_decay: float = 1e-4 # optimizer weight decay

    ##### warmup #####
    warmup_epochs: float = 2.0 # warmup epochs (fractions ok)
    warmup_momentum: float = 0.8 # warmup initial momentum
    warmup_bias_lr: float = 6e-4 # warmup initial bias learning rate

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config parameter: {k}.")