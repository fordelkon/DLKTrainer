from dataclasses import dataclass, field
from typing import List

from ...trainer_config import BaseTrainerConfig


@dataclass
class VQVAEnTrainerConfig(BaseTrainerConfig):
    ##### base config #####
    save_dir: str = "runs_vae" # directory to save the experiment results
    batch_size: int = 16 # batch size 
    nominal_batch_size: int = 64 # nominal batch size
    num_workers: int = 8 # how many subprocesses are used to load data in parallel
    epochs: int = 600 # number of epochs to train
    validation_epochs: int = 200 # do validation inference at `validation_epochs` times
    show_image_epochs: int = 30 # show validation images at `show_image_epochs` times
    amp: bool = False # whether to use automatic mixed precision training
    split_ratios: List[float] = field(default_factory=lambda: [0.8, 0.2])  

    ##### optimizer & scheduler #####
    optimizer: str = "Adam" # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp]
    scheduler: str = "MultiCyclic" # learning rate scheduler to use, choices=[Cosine, Linear, Exponential, Polynomial, Constant, "MultiStep", "MultiCyclic"]
    n_cycles: int = 3 # number of cycles to run for multi-cyclic 
    cycle_scheduler: List[str] = field(default_factory=lambda: ["Cosine"] * 3) # type(s) of scheduler function used within each cycle (e.g., "Cosine", "Linear", "Constant", "Polynominal", "Exponential"). 
    initial_lr: float = 3e-4 # initial learning rate
    f2i_lr: float = 0.01 # final_lr = initial_lr * f2i_lr
    momentum: float = 0.937 # SGD momentum/Adam beta1
    weight_decay: float = 1e-4 # optimizer weight decay

    ##### warmup #####
    warmup_epochs: float = 25.0 # warmup epochs (fractions ok)
    warmup_momentum: float = 0.8 # warmup initial momentum
    warmup_bias_lr: float = 6e-4 # warmup initial bias learning rate

    ##### vqvae2 config #####
    recon_weight: float = 1.
    latent_weight: float = 0.25