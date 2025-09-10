# See the detailed mathematical derivation about original GAN in https://fordelkon.github.io/posts/prob_dl/#-mathrmgangenerativeadversarialnetworks-

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Literal, Dict, Tuple, Union, List
import math
import warnings
import gc

from ...trainer import BaseTrainer
from .trainer_config import DCGANTrainerConfig
from ...utils.trainer_utils import select_device, check_amp


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DCGANTrainer(BaseTrainer):
    def __init__(self, model: Tuple[nn.Module], dataset: Dataset, cfg: DCGANTrainerConfig):
        super().__init__(model, dataset)
        # model[0]: generator, model[1]: discriminator
        self.mapping = {"generator": 0, "discriminator": 1}
        self.model = model
        self.dataset = dataset

        ##### Basic trainer config #####
        self.device = select_device(cfg.device)
        self.batch_size = cfg.batch_size
        self.nominal_batch_size = cfg.nominal_batch_size
        self.num_workers = cfg.num_workers
        self.epochs = cfg.epochs

        self.latent_dim = cfg.latent_dim

        ##### Optimizer & Scheduler config (generator) #####
        self.optimizer_choice_g = cfg.optimizer_g
        self.scheduler_choice_g = cfg.scheduler_g
        self.momentum_g = cfg.momentum_g
        self.weight_decay_g = cfg.weight_decay_g
        self.warmup_epochs_g = cfg.warmup_epochs_g
        self.warmup_bias_lr_g = cfg.warmup_bias_lr_g
        self.warmup_momentum_g = cfg.warmup_momentum_g
        self.initial_lr_g = cfg.initial_lr_g

        self.scheduler_params_g = {
            "f2i_lr": cfg.f2i_lr, 
            "gamma": cfg.gamma, 
            "power": cfg.power, 
            "milestones": cfg.milestones, 
            "step_ratio": cfg.step_ratio, 
            "n_cycles": cfg.n_cycles, 
            "cycle_scheduler": cfg.cycle_scheduler
        }

        ##### Optimizer & Scheduler config (discriminator) #####
        self.optimizer_choice_d = cfg.optimizer_d
        self.scheduler_choice_d = cfg.scheduler_d
        self.momentum_d = cfg.momentum_d
        self.weight_decay_d = cfg.weight_decay_d
        self.warmup_epochs_d = cfg.warmup_epochs_d
        self.warmup_bias_lr_d = cfg.warmup_bias_lr_d
        self.warmup_momentum_d = cfg.warmup_momentum_d
        self.initial_lr_d = cfg.initial_lr_d

        self.scheduler_params_d = {
            "f2i_lr": cfg.f2i_lr, 
            "gamma": cfg.gamma, 
            "power": cfg.power, 
            "milestones": cfg.milestones, 
            "step_ratio": cfg.step_ratio, 
            "n_cycles": cfg.n_cycles, 
            "cycle_scheduler": cfg.cycle_scheduler
        }

        ##### Trainer way config #####
        self.amp = cfg.amp

        ##### Extra class attribute #####
        self.loader = self.get_loader(dataset)
        self.save_dir = Path(cfg.save_dir)
        self.validation_epochs = cfg.validation_epochs
        self.show_image_epochs = cfg.show_image_epochs
        self.current_epoch = 0 # The model is training in current epoch
        self.accumulate_g = max(round(self.nominal_batch_size / self.batch_size), 1) # accumulate loss before optimizing
        self.accumulate_d = max(round(self.nominal_batch_size / self.batch_size), 1) # accumulate loss before optimizing

    def _setup_optimizer(self, model_choice: Literal["generator", "discriminator"], 
                                        choice: Literal["Adam", "Adamax", "AdamW", "NAdam", 
                                              "RAdam", "RMSProp", "SGD"]="Adam",
                                              initial_lr: float=1e-4, momentum: float=0.937, weight_decay: float=5e-4):
        g = [], [], [], [] # optimizer parameter groups
        n = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k) # normalization layers, i.e. BatchNorm2d
        e = tuple(v for k, v in torch.nn.__dict__.items() if "Embedding" in k) # embedding layers, i.e. Embedding
        for module_name, module in self.model[self.mapping[model_choice]].named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname: # bias (no decay)
                    g[3].append(param)
                elif isinstance(module, e): # weight (less decay)
                    g[2].append(param)
                elif isinstance(module, n) or "logit_scale" in fullname: # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1].append(param)
                else: # weight (with decay)
                    g[0].append(param)
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD"}

        if choice in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(torch.optim, choice, torch.optim.Adam)(g[3], lr=initial_lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif choice == "RMSProp":
            optimizer = torch.optim.RMSprop(g[3], lr=initial_lr, momentum=momentum)
        elif choice == "SGD":
            optimizer = torch.optim.SGD(g[3], lr=initial_lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"optimizer '{choice}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )
        
        optimizer.add_param_group({"params": g[0], "weight_decay": weight_decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        optimizer.add_param_group({"params": g[2], "weight_decay": weight_decay * 1e-2}) # add g2 (Embedding weight)

        logging.info(
            f"'{model_choice} optimizer:' {type(optimizer).__name__}(lr={initial_lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[2])} weight(decay={weight_decay*1e-2}), {len(g[0])} weight(decay={weight_decay}), {len(g[3])} bias(decay=0.0)"
        )
        return optimizer
    
    def _get_base_scheduler(self, scheduler_type: Literal["Cosine", "Linear", "Exponential", 
                                            "Polynomial", "Constant"], cycle_progress: float, scheduler_params: Dict[str, Union[float, int]]):
        if scheduler_type == "Cosine":
            return max((1 - math.cos(cycle_progress * math.pi)) / 2, 0) * (scheduler_params["f2i_lr"] - 1) + 1
        elif scheduler_type == "Linear":
            return max(1 - cycle_progress, 0) * (1 - scheduler_params["f2i_lr"]) + scheduler_params["f2i_lr"]
        elif scheduler_type == "Exponential":
            cycle_epochs = self.epochs / scheduler_params["n_cycles"]
            exp_factor = cycle_epochs * cycle_progress
            return scheduler_params["gamma"] ** max(exp_factor, 0) ** scheduler_params["power"] * (1 - scheduler_params["f2i_lr"]) + scheduler_params["f2i_lr"]
        elif scheduler_type == "Polynomial":
            return max(1 - cycle_progress, 0) ** scheduler_params["power"] * (1 - scheduler_params["f2i_lr"]) + scheduler_params["f2i_lr"]
        elif scheduler_type == "Constant":
            return 1.0
        else:
            raise ValueError(f"MultiCyclic Scheduler can't support base scheduler type: {scheduler_type}")
    
    def _setup_scheduler(self, optimizer: Optimizer, choice: Literal["Cosine", "Linear", "Exponential", 
                                               "Polynomial", "Constant", "MultiStep", "MultiCyclic"]="Cosine", 
                         scheduler_params: Dict[str, Union[float, List[float], str]]={
                             "f2i_lr": 0.1, 
                             "gamma": 0.9, 
                             "power": 2.0,
                             "milestones": [50, 100, 150], 
                             "step_ratio": 0.1, 
                             "n_cycles": 1, 
                             "cycle_scheduler": ["Cosine"], 
                         }):
        if choice == "Cosine":
            self.lr_func = lambda x: max((1 - math.cos(x * math.pi / self.epochs)) / 2, 0) * (scheduler_params["f2i_lr"] - 1) + 1 # Cosine
        elif choice == "Linear":
            self.lr_func = lambda x: max(1 - x / self.epochs, 0) * (1 - scheduler_params["f2i_lr"]) + scheduler_params["f2i_lr"] # Linear
        elif choice == "Exponential":
            self.lr_func = lambda x: scheduler_params["gamma"] ** max(x, 0) * (1 - scheduler_params["f2i_lr"]) + scheduler_params["f2i_lr"] # Exponential
        elif choice == "Polynomial":
            self.lr_func = lambda x: max(1 - x / self.epochs, 0) ** scheduler_params["power"] * (1 - scheduler_params["f2i_lr"]) + scheduler_params["f2i_lr"] # Polynomial
        elif choice == "Constant":
            self.lr_func = lambda x: 1. # Constant
        elif choice == "MultiStep":
            self.lr_func = lambda x: scheduler_params["step_ratio"] ** sum(1 for milestone in scheduler_params["milestones"] if x >= milestone)
        elif choice == "MultiCyclic":
            def multicyclic_lr_func(x):
                cycle_len = self.epochs / scheduler_params["n_cycles"]
                cycle_progress = (x % cycle_len) / cycle_len
                milestones = [cycle_len * i for i in range(1, scheduler_params["n_cycles"])]
                idx = sum(1 for milestone in milestones if x >= milestone - 1)
                return self._get_base_scheduler(scheduler_type=scheduler_params["cycle_scheduler"][idx], 
                                                cycle_progress=cycle_progress, scheduler_params=scheduler_params)
            self.lr_func = multicyclic_lr_func
        else:
            raise ValueError("The scheduler type is not supported!!!")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_func)
        return scheduler
    
    def _setup_train(self):
        # Set up Optimizer and Scheduler for generator and discriminator
        scaled_weight_decay_g = self.weight_decay_g * self.batch_size * self.accumulate_g / self.nominal_batch_size # scale weight decay for generator
        scaled_weight_decay_d = self.weight_decay_d * self.batch_size * self.accumulate_d / self.nominal_batch_size # scale weight decay for discriminator
        self.optimizer_g = self._setup_optimizer(
            model_choice="generator",
            choice=self.optimizer_choice_g, 
            initial_lr=self.initial_lr_g, 
            momentum=self.momentum_g, 
            weight_decay=scaled_weight_decay_g
        )
        self.optimizer_d = self._setup_optimizer(
            model_choice="discriminator",
            choice=self.optimizer_choice_d, 
            initial_lr=self.initial_lr_d, 
            momentum=self.momentum_d, 
            weight_decay=scaled_weight_decay_d
        )

        self.scheduler_g = self._setup_scheduler(
            optimizer=self.optimizer_g, 
            choice=self.scheduler_choice_g, 
            scheduler_params=self.scheduler_params_g
        )
        self.scheduler_d = self._setup_scheduler(
            optimizer=self.optimizer_d, 
            choice=self.scheduler_choice_d, 
            scheduler_params=self.scheduler_params_d
        )
        self.scheduler_g.last_epoch = self.current_epoch - 1 # should start from -1, and auto increase with epoch
        self.scheduler_d.last_epoch = self.current_epoch - 1 # should start from -1, and auto increase with epoch

    def _optimizer_step(self, optimizer: Optimizer, 
                        model_choice: Literal["generator", "discriminator"]):      
        self.scaler.unscale_(optimizer) # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model[self.mapping[model_choice]].parameters(), max_norm=10.) # clip gradients
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad()

    def train(self):
        self._setup_train()

        self.generator = self.model[0].to(self.device)
        self.discriminator = self.model[1].to(self.device)

        if self.amp:
            self.amp = check_amp(self.model[0]) and check_amp(self.model[1]) # generator
        self.scaler = torch.amp.GradScaler(self.device, enabled=self.amp)

        num_batches = len(self.loader)
        num_warmup_iters_g = max(round(self.warmup_epochs_g * num_batches), 100) if self.warmup_epochs_g > 0 else -1
        num_warmup_iters_d = max(round(self.warmup_epochs_d * num_batches), 100) if self.warmup_epochs_d > 0 else -1
        last_opt_step_g = -1
        last_opt_step_d = -1
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        
        valid = torch.ones(size=(self.batch_size,), device=self.device)
        fake = torch.zeros(size=(self.batch_size,), device=self.device)

        for epoch in range(self.epochs):
            logging.info("-" * 20 + "\n")
            self.current_epoch = epoch
            train_loss_g = 0.
            train_loss_d = 0.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler_g.step()
                self.scheduler_d.step()
            self.generator.train()
            self.discriminator.train()
            pbar = tqdm(enumerate(self.loader), total=num_batches, bar_format="{l_bar}{bar:10}{r_bar}")

            for i, batch in pbar:
                ##### Train Generator #####
                current_num_iter = i + num_batches * epoch
                # Warmup Generator
                if current_num_iter <= num_warmup_iters_g:
                    x_interp_g = [0, num_warmup_iters_g]
                    # gradient accumulation scheduling, accmulate <-> iter to update parameters
                    self.accumulate_g = max(1, int(np.interp(current_num_iter, x_interp_g, [1, self.nominal_batch_size / self.batch_size]).round()))
                    for j, param_group in enumerate(self.optimizer_g.param_groups):
                        # Bias lr falls from warmup_bias_lr to ～initial_lr, all other lrs rise from 0.0 to ～initial_lr (more smoothing)
                        param_group["lr"] = np.interp(
                            current_num_iter, x_interp_g, [self.warmup_bias_lr_g if j == 0 else 0.0, param_group["initial_lr"] * self.lr_func(self.current_epoch)]
                        )
                        if "momentum" in param_group:
                            param_group["momentum"] = np.interp(current_num_iter, x_interp_g, [self.warmup_momentum_g, self.momentum_g])

                # Generator Forward
                Z = torch.randn(size=(self.batch_size, self.latent_dim), device=self.device)
                imgen = self.generator(Z)
                loss_g = nn.functional.binary_cross_entropy(self.discriminator(imgen), valid, reduction="mean")
                train_loss_g += loss_g.item()

                # Generator Backward
                self.scaler.scale(loss_g).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if current_num_iter - last_opt_step_g >= self.accumulate_g:
                    self._optimizer_step(self.optimizer_g, model_choice="generator")
                    last_opt_step_g = current_num_iter

                
                ##### Train Discriminator #####
                # Warmup Discriminator
                if current_num_iter <= num_warmup_iters_d:
                    x_interp_d = [0, num_warmup_iters_d]
                    # gradient accumulation scheduling, accmulate <-> iter to update parameters
                    self.accumulate_d = max(1, int(np.interp(current_num_iter, x_interp_d, [1, self.nominal_batch_size / self.batch_size]).round()))
                    for j, param_group in enumerate(self.optimizer_d.param_groups):
                        # Bias lr falls from warmup_bias_lr to ～initial_lr, all other lrs rise from 0.0 to ～initial_lr (more smoothing)
                        param_group["lr"] = np.interp(
                            current_num_iter, x_interp_d, [self.warmup_bias_lr_d if j == 0 else 0.0, param_group["initial_lr"] * self.lr_func(self.current_epoch)]
                        )
                        if "momentum" in param_group:
                            param_group["momentum"] = np.interp(current_num_iter, x_interp_d, [self.warmup_momentum_d, self.momentum_d])

                # Discriminator Forward
                X = batch[0].to(self.device)
                loss_d1 = nn.functional.binary_cross_entropy(self.discriminator(X), valid, reduction="mean")
                loss_d2 = nn.functional.binary_cross_entropy(self.discriminator(imgen.detach()), fake, reduction="mean")
                loss_d = (loss_d1 + loss_d2) / 2
                train_loss_d += loss_d.item()

                # Discriminator Backward
                self.scaler.scale(loss_d).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if current_num_iter - last_opt_step_d >= self.accumulate_d:
                    self._optimizer_step(self.optimizer_d, model_choice="discriminator")
                    last_opt_step_d = current_num_iter

                pbar.set_description(
                    f"{self.current_epoch}/{self.epochs}", 
                    f"{self.get_memory():.3g}G", # (GB) GPU memory util
                )
            
            lr_g = {f"lr/param_group{k}": param_group["lr"] for k, param_group in enumerate(self.optimizer_g.param_groups)}
            lr_d = {f"lr/param_group{k}": param_group["lr"] for k, param_group in enumerate(self.optimizer_d.param_groups)}

            logging.info(f"Generator: all types `lr` of epoch {self.current_epoch}: {lr_g}\n"
                         "- lr/param_group0: regular weights (full weight decay applied)\n"
                         "- lr/param_group1: batchnorm and logit_scale parameters (no weight decay)\n"
                         "- lr/param_group2: embedding layer weights (smaller weight decay)\n"
                         "- lr/param_group3: bias parameters (no weight decay)")
            
            train_loss_g /= num_batches
            logging.info(f"epoch {self.current_epoch}: generator train loss {train_loss_g}")

            logging.info(f"Discriminator: all types `lr` of epoch {self.current_epoch}: {lr_d}\n"
                         "- lr/param_group0: regular weights (full weight decay applied)\n"
                         "- lr/param_group1: batchnorm and logit_scale parameters (no weight decay)\n"
                         "- lr/param_group2: embedding layer weights (smaller weight decay)\n"
                         "- lr/param_group3: bias parameters (no weight decay)")
            train_loss_d /= num_batches
            logging.info(f"epoch {self.current_epoch}: discriminator train loss {train_loss_d}")

            if (self.current_epoch + 1) % self.validation_epochs == 0 or (self.current_epoch + 1) % self.show_image_epochs == 0:
                self.save_dir.mkdir(exist_ok=True)
                X_sampled = self.validate()
                if (self.current_epoch + 1) % self.validation_epochs == 0:
                    self.checkpoint = self.save_dir / f"checkpoint-{self.current_epoch + 1}.pt"
                    self.save_model()
                if (self.current_epoch + 1) % self.show_image_epochs == 0:
                    self.im_path = self.save_dir / f"image-{self.current_epoch + 1}.jpg"
                    self.save_images(X_sampled)

            if self.get_memory(fraction=True) > 0.5:
                self.clear_memory() # clear if memory utilizaiton > 50%
            
            logging.info(
                f"{self.current_epoch} epochs forward process completed!\n"
            )
            logging.info("-" * 20 + "\n")
        self.clear_memory()

    @torch.inference_mode()
    def validate(self):
        self.generator.eval()
        Z = torch.randn(size=(self.batch_size, self.latent_dim), device=self.device)
        imgen = self.generator(Z)
        self.generator.train()
        imgen = (imgen.clamp(-1, 1) + 1) / 2
        imgen = (imgen * 255).to(torch.uint8)
        logging.info(f"{self.current_epoch} epoch generator generate images complete!")
        return imgen
    
    def get_loader(self, dataset: Dataset, mode: str="train"):
        shuffle = mode == "train"
        loader = DataLoader(dataset, batch_size=self.batch_size, 
                            num_workers=self.num_workers, shuffle=shuffle, drop_last=True)
        return loader

    def get_memory(self, fraction=False):
        """
        Get the current GPU memory usage.

        This method queries the GPU memory usage status and can return either the absolute memory usage 
        or the usage as a fraction of total GPU memory. For CPU training, it returns 0.

        Args:
            fraction (bool): Whether to return memory usage as a fraction.
                - True: Returns usage as a fraction (between 0 and 1).
                - False: Returns absolute memory usage in GB.

        Returns:
            float: Memory usage.
                - If fraction=True, returns a value between 0 and 1 representing the fraction used.
                - If fraction=False, returns the absolute memory usage in gigabytes.
                - Returns 0 if device is CPU.

        Note:
            - Only applicable for CUDA devices; CPU devices always return 0.
            - Uses torch.cuda.memory_reserved() to get reserved memory.
            - Useful for monitoring memory usage during training.
        """
        memory, total = 0, 0
        if self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def clear_memory(self):
        """
        Clear memory caches.

        This method performs memory cleanup to free unused memory:
        1. Calls the Python garbage collector to clean up CPU memory.
        2. If using a GPU, clears the CUDA cache.

        Note:
            - Calling this periodically during training can prevent memory overflow.
            - Especially useful when handling large models or large batch sizes.
            - In CPU mode, only garbage collection is performed.
        """
        gc.collect()
        if self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()   

    def save_model(self):
        """
        Save the current model checkpoint to disk.
        
        This method serializes the training state including:
        - Current epoch
        - Best fitness score achieved so far
        - Model weights (supports both DataParallel and normal models)
        - Optimizer state
        - Timestamp of saving

        The checkpoint is saved to 'last.pt', and if the current model has the best fitness,
        it is also saved to 'best.pt'.
        """
        import io
        from copy import deepcopy
        from datetime import datetime

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.current_epoch,
                "model": deepcopy(self.generator), # support parallelization 
                "optimizer": deepcopy(self.optimizer_g.state_dict()),
                "date": datetime.now().isoformat(),
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        logging.info(f"epoch {self.current_epoch} has been saved")
        self.checkpoint.write_bytes(serialized_ckpt)  # save last.pt

    def save_images(self, images, **kwargs):
        # images: [0,1] float
        grid = torchvision.utils.make_grid(images, **kwargs)  
        array = grid.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(array)
        image.save(self.im_path)     