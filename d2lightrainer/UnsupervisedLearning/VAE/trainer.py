# See the detailed mathematical derivation about original VAE in https://fordelkon.github.io/posts/prob_dl/#-mathrmvaevaritionalautoencoders-

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Optimizer
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Literal, Dict, Tuple, List, Union
import math
import warnings
import gc
import itertools
import random

from ...trainer import BaseTrainer
from .trainer_config import VAETrainerConfig
from ...utils.trainer_utils import select_device, check_amp


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class VAETrainer(BaseTrainer):
    def __init__(self, model: Tuple[nn.Module], dataset: Dataset, cfg: VAETrainerConfig):
        super().__init__(model, dataset)
        # model[0]: encoder, model[1]: decoder
        self.encoder = model[0]
        self.decoder = model[1]
        self.dataset = dataset

        ##### Basic trainer config #####
        self.device = select_device(cfg.device)
        self.batch_size = cfg.batch_size
        self.nominal_batch_size = cfg.nominal_batch_size
        self.num_workers = cfg.num_workers
        self.epochs = cfg.epochs

        ##### Warmup trainer config #####
        self.warmup_epochs = cfg.warmup_epochs
        self.warmup_bias_lr = cfg.warmup_bias_lr
        self.warmup_momentum = cfg.warmup_momentum

        ##### Optimizer & Scheduler config #####
        self.optimizer_choice = cfg.optimizer
        self.scheduler_choice = cfg.scheduler
        self.momentum = cfg.momentum
        self.weight_decay = cfg.weight_decay

        self.initial_lr = cfg.initial_lr
        self.scheduler_params = {
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
        self.split_ratios = cfg.split_ratios
        if len(self.split_ratios) == 2:
            self.train_dataset, self.val_dataset = self.split_dataset(
                self.split_ratios
            ) 
            self.train_loader, self.val_loader = self.get_loader(self.train_dataset), self.get_loader(self.val_dataset, mode="notrain")
        elif len(self.split_ratios) == 3:
            self.train_dataset, self.val_dataset, self.test_dataset = self.split_dataset(
                self.split_ratios
            ) 
            self.train_loader, self.val_loader, self.test_loader = self.get_loader(self.train_dataset), self.get_loader(self.val_dataset, mode="notrain"), self.get_loader(self.test_dataset, mode="notrain")
        else:
            raise ValueError("You can only split the dataset into 2 or 3 parts!!!")
        
        self.save_dir = Path(cfg.save_dir)
        self.validation_epochs = cfg.validation_epochs
        self.show_image_epochs = cfg.show_image_epochs
        self.current_epoch = 0 # The model is training in current epoch
        self.accumulate = max(round(self.nominal_batch_size / self.batch_size), 1) # accumulate loss before optimizing

        ##### vae config #####
        self.recon_weight = cfg.recon_weight
        self.kl_weight = cfg.kl_weight

    def split_dataset(self, split_ratios: List[float]):
        assert sum(split_ratios) == 1.
        if len(split_ratios) == 2:
            train_size = int(split_ratios[0] * len(self.dataset))
            val_size = len(self.dataset) - train_size
            return random_split(self.dataset, [train_size, val_size])
        if len(split_ratios) == 3:
            train_size = int(split_ratios[0] * len(self.dataset))
            val_size = int(split_ratios[1] * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size
            return random_split(self.dataset, [train_size, val_size, test_size])

    def _setup_optimizer(self, choice: Literal["Adam", "Adamax", "AdamW", "NAdam", 
                                              "RAdam", "RMSProp", "SGD"]="Adam",
                                              initial_lr: float=1e-4, momentum: float=0.937, weight_decay: float=5e-4):
        g = [], [], [], [] # optimizer parameter groups
        n = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k) # normalization layers, i.e. BatchNorm2d
        e = tuple(v for k, v in torch.nn.__dict__.items() if "Embedding" in k) # embedding layers, i.e. Embedding
        named_modules = itertools.chain(self.encoder.named_modules(), self.decoder.named_modules())
        for module_name, module in named_modules:
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
            f"'optimizer:' {type(optimizer).__name__}(lr={initial_lr}, momentum={momentum}) with parameter groups "
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
    
    def _setup_scheduler(self, choice: Literal["Cosine", "Linear", "Exponential", 
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
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_func)
        return scheduler
    
    def _setup_train(self):
        # Set up Optimizer and Scheduler
        scaled_weight_decay = self.weight_decay * self.batch_size * self.accumulate / self.nominal_batch_size # scale weight decay
        self.optimizer = self._setup_optimizer(
            choice=self.optimizer_choice, 
            initial_lr=self.initial_lr, 
            momentum=self.momentum, 
            weight_decay=scaled_weight_decay
        )
        self.scheduler = self._setup_scheduler(
            choice=self.scheduler_choice, 
            scheduler_params=self.scheduler_params
        )
        self.scheduler.last_epoch = self.current_epoch - 1 # should start from -1, and auto increase with epoch

    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer) # unscale gradients
        parameters = itertools.chain(self.encoder.parameters(), self.decoder.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.) # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def train(self):
        self._setup_train()

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        if self.amp:
            self.amp = check_amp(self.encoder) and check_amp(self.decoder)
        self.scaler = torch.amp.GradScaler(self.device, enabled=self.amp)

        num_batches = len(self.train_loader) # number of batches
        num_warmup_iters = max(round(self.warmup_epochs * num_batches), 100) if self.warmup_epochs > 0 else -1 # warmup iterations
        last_opt_step = -1
        self.optimizer.zero_grad()
        for epoch in range(self.epochs):
            logging.info("-" * 20 + "\n")
            train_loss = 0.
            self.current_epoch = epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
            self.encoder.train()
            self.decoder.train()
            pbar = tqdm(enumerate(self.train_loader), total=num_batches, bar_format="{l_bar}{bar:10}{r_bar}")

            for i, batch in pbar:
                # Warmup 
                current_num_iter = i + num_batches * epoch
                if current_num_iter <= num_warmup_iters:
                    x_interp = [0, num_warmup_iters]
                    # gradient accumulation scheduling, accmulate <-> iter to update parameters
                    self.accumulate = max(1, int(np.interp(current_num_iter, x_interp, [1, self.nominal_batch_size / self.batch_size]).round()))
                    for j, param_group in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from warmup_bias_lr to ～initial_lr, all other lrs rise from 0.0 to ～initial_lr (more smoothing)
                        param_group["lr"] = np.interp(
                            current_num_iter, x_interp, [self.warmup_bias_lr if j == 0 else 0.0, param_group["initial_lr"] * self.lr_func(self.current_epoch)]
                        )
                        if "momentum" in param_group:
                            param_group["momentum"] = np.interp(current_num_iter, x_interp, [self.warmup_momentum, self.momentum])
                
                # Forward
                X = batch[0].to(self.device)
                mu, log_var = self.encoder(X)
                # Reparameterization
                std = torch.exp(0.5 * log_var)
                epsilon = torch.randn_like(std)
                Z = mu + std * epsilon
                # Reparameterization
                X_hat = self.decoder(Z)
                loss_recon = nn.functional.mse_loss(X, X_hat, reduction="mean")
                loss_kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=1), dim=0)
                loss = self.recon_weight * loss_recon + self.kl_weight * loss_kl
                train_loss += loss.item()

                # Backward
                self.scaler.scale(loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if current_num_iter - last_opt_step >= self.accumulate:
                    self._optimizer_step()
                    last_opt_step = current_num_iter

                pbar.set_description(
                    f"{self.current_epoch}/{self.epochs}",
                    f"{self.get_memory():.3g}G",  # (GB) GPU memory util
                )

            lr = {f"lr/param_group{k}": param_group["lr"] for k, param_group in enumerate(self.optimizer.param_groups)}
            logging.info(f"All types `lr` of epoch {self.current_epoch}: {lr}\n"
                         "- lr/param_group0: regular weights (full weight decay applied)\n"
                         "- lr/param_group1: batchnorm and logit_scale parameters (no weight decay)\n"
                         "- lr/param_group2: embedding layer weights (smaller weight decay)\n"
                         "- lr/param_group3: bias parameters (no weight decay)")

            train_loss /= num_batches
            logging.info(f"epoch {self.current_epoch}: train loss {train_loss}")

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
                f"{self.current_epoch} epochs completed!\n"
            )
            logging.info("-" * 20 + "\n")
        self.clear_memory()

    @torch.inference_mode()
    def validate(self):
        num_batches = len(self.val_loader)
        rand_batch = random.randint(0, num_batches - 1)
        self.encoder.eval()
        self.decoder.eval()
        pbar = tqdm(enumerate(self.val_loader), total=num_batches, bar_format="{l_bar}{bar:10}{r_bar}")
        val_loss = 0.
        X_sampled = None

        for i, batch in pbar:
            # Forward
            X = batch[0].to(self.device)
            mu, log_var = self.encoder(X)
            # Reparameterization
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std)
            Z = mu + std * epsilon
            # Reparameterization
            X_hat = self.decoder(Z)
            if i == rand_batch:
                X_sampled = X_hat
            loss_recon = nn.functional.mse_loss(X, X_hat, reduction="mean")
            loss_kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=1), dim=0)
            loss = self.recon_weight * loss_recon + self.kl_weight * loss_kl
            val_loss += loss.item()
        
        val_loss /= num_batches
        logging.info(f"epoch {self.current_epoch}: val loss {val_loss}")
        self.encoder.train()
        self.decoder.train()
        X_sampled = (X_sampled.clamp(-1, 1) + 1) / 2
        X_sampled = (X_sampled * 255).to(torch.uint8)
        logging.info(f"{self.current_epoch} epoch vae reconstruct images complete!")
        return X_sampled

    def get_loader(self, dataset: Dataset, mode: str="train"):
        shuffle = mode == "train"
        loader = DataLoader(dataset, batch_size=self.batch_size, 
                            num_workers=self.num_workers, shuffle=shuffle)
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
                "encoder": deepcopy(self.encoder), # 
                "decoder": deepcopy(self.decoder), # 
                "optimizer": deepcopy(self.optimizer.state_dict()),
                "date": datetime.now().isoformat(),
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        logging.info(f"epoch {self.current_epoch} has been saved")
        self.checkpoint.write_bytes(serialized_ckpt)  # save checkpoint.pt

    def save_images(self, images, **kwargs):
        # images: [0,1] float
        grid = torchvision.utils.make_grid(images, **kwargs)  
        array = grid.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(array)
        image.save(self.im_path)     