from dataclasses import dataclass

from ...trainer_config import BaseTrainerConfig


@dataclass
class DDPMTrainerConfig(BaseTrainerConfig):
    ##### base config #####
    save_dir: str = "runs_dm" # directory to save the experiment results
    batch_size: int = 16 # batch size 
    nominal_batch_size: int = 64 # nominal batch size
    num_workers: int = 8 # how many subprocesses are used to load data in parallel
    epochs: int = 600 # number of epochs to train
    validation_epochs: int = 200 # do validation inference at `validation_epochs` times
    show_image_epochs: int = 20 # show validation images at `show_image_epochs` times
    amp: bool = False # whether to use automatic mixed precision training

    ##### optimizer & scheduler #####
    optimizer: str = "Adam" # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp]
    scheduler: str = "Cosine" # learning rate scheduler to use, choices=[Cosine, Linear, Exponential, Polynomial, Constant]
    initial_lr: float = 3e-4 # initial learning rate
    f2i_lr: float = 0.1 # final_lr = initial_lr * f2i_lr
    momentum: float = 0.937 # SGD momentum/Adam beta1
    weight_decay: float = 1e-4 # optimizer weight decay

    ##### warmup #####
    warmup_epochs: float = 20.0 # warmup epochs (fractions ok)
    warmup_momentum: float = 0.8 # warmup initial momentum
    warmup_bias_lr: float = 6e-4 # warmup initial bias learning rate

    ##### ddpm config #####
    noise_step: int = 1000
    denoise_step: int = 1000
    beta_start: float = 1e-4
    beta_end: float=0.02
    img_size: int=64