from dataclasses import dataclass

from ...trainer_config import BaseTrainerConfig


@dataclass
class DCGANTrainerConfig(BaseTrainerConfig):
    ##### base config #####
    save_dir: str = "runs_dcgan" # directory to save the experiment results
    batch_size: int = 16 # batch size
    nominal_batch_size: int = 64 # nominal batch size
    num_workers: int = 8 # how many subprocesses are used to load data in parallel
    epochs: int = 200 # number of epochs to train
    validation_epochs: int = 50 # do validation inference at `validation_epochs` times
    show_image_epochs: int = 10 # show validation images at `show_image_epochs` times
    amp: bool = False # whether to use automatic mixed precision training 

    ##### optimizer & scheduler (generator) #####
    optimizer_g: str = "Adam" # optimizer to use for generator, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp]
    scheduler_g: str = "Constant" # learning rate scheduler to use for generator, choices=[Cosine, Linear, Exponential, Polynomial, Constant]
    initial_lr_g: float = 3e-4 # initial learning rate of generator
    momentum_g: float = 0.500 # SGD momentum/Adam beta1 for generator
    weight_decay_g: float = 0. # optimizer weight decay for generator
    warmup_epochs_g: float = 5.0 # warmup epochs (fractions ok) of generator
    warmup_momentum_g: float = 0.0 # warmup initial momentum of generator
    warmup_bias_lr_g: float = 6e-4 # warmup initial bias learning rate of generator

    ##### optimizer & scheduler (descriminator) #####
    optimizer_d: str = "Adam" # optimizer to use for descriminator, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp]
    scheduler_d: str = "Constant" # learning rate scheduler to use for descriminator, choices=[Cosine, Linear, Exponential, Polynomial, Constant]
    initial_lr_d: float = 3e-4 # initial learning rate of descriminator
    momentum_d: float = 0.500 # SGD momentum/Adam beta1 for descriminator
    weight_decay_d: float = 1e-4 # optimizer weight decay for descriminator
    warmup_epochs_d: float = 2.0 # warmup epochs (fractions ok) of descriminator
    warmup_momentum_d: float = 0.0 # warmup initial momentum of descriminator
    warmup_bias_lr_d: float = 6e-4 # warmup initial bias learning rate of descriminator

    ##### dcgan config #####
    latent_dim: int = 100