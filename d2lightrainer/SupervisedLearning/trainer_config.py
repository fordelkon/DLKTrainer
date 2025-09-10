from dataclasses import dataclass, field
from typing import List

from ..trainer_config import BaseTrainerConfig


@dataclass
class CLSTrainerConfig(BaseTrainerConfig):
    ##### base config #####
    save_dir: str = "runs_cls"  # directory to save the experiment results
    batch_size: int = 16  # batch size 
    nominal_batch_size: int = 64  # nominal batch size
    num_workers: int = 8  # how many subprocesses are used to load data in parallel
    epochs: int = 200  # number of epochs to train
    amp: bool = False  # whether to use automatic mixed precision training
    split_ratios: List[float] = field(default_factory=lambda: [0.8, 0.2])  # ✅ 修复
    patience: int = 50  # early stopping tolerance 
    stop_ratio: float = 1.  # ratio of stopping metrics

    ##### optimizer & scheduler #####
    optimizer: str = "Adam" # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp]
    scheduler: str = "Cosine" # learning rate scheduler to use, choices=[Cosine, Linear, Exponential, Polynomial, Constant]
    initial_lr: float = 3e-4 # initial learning rate
    f2i_lr: float = 0.1 # final_lr = initial_lr * f2i_lr
    momentum: float = 0.937 # SGD momentum/Adam beta1
    weight_decay: float = 1e-4 # optimizer weight decay

    ##### warmup #####
    warmup_epochs: float = 5.0 # warmup epochs (fractions ok)
    warmup_momentum: float = 0.8 # warmup initial momentum
    warmup_bias_lr: float = 6e-4 # warmup initial bias learning rate