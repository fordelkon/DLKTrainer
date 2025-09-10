import torch
from typing import Dict, Optional
import logging
import re


def select_device(device: int):
    """
    Select single computing device (GPU/CPU).
    
    Args:
        device (int): GPU ID to use. If negative or GPU is unavailable, CPU will be used.
        
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        if device >= torch.cuda.device_count() or device < 0:
            logging.info(f"Using GPU: 0")
            return torch.device("cuda:0")
        else:
            logging.info(f"Using GPU: {device}")
            return torch.device(f"cuda:{device}")
    else:
        logging.info("Using CPU")
        return torch.device("cpu")


def check_amp(model: torch.nn.Module):
    """Check if Automatic Mixed Precision (AMP) training is safe to use on the given model's GPU.

    Some GPUs (e.g., GTX 16xx series and certain Quadros/Teslas) are known to cause instability 
    when using AMP (e.g., NaN losses or zero mAP). This function detects such GPUs and disables AMP accordingly.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model whose parameters will be checked to determine the device type (CPU or GPU).

    Returns
    -------
    bool
        True if AMP is safe to use on the detected device.
        False if AMP should be disabled (e.g., on problematic GPUs or CPU).

    Examples
    --------
    >>> model = MyModel().to('cuda:0')
    >>> amp_ok = check_amp(model)
    >>> print(f"AMP usable: {amp_ok}")
    """
    logging.info("Check whether AMP(Automatic Mixed Precision) training is safe...")
    device = next(model.parameters()).device
    if device.type == "cpu": 
        logging.warning("The model is training on CPU, it can't be trained with AMP!!!")
        return False
    else:
        # GPUs that have issues with AMP
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", re.IGNORECASE
        )

        gpu = torch.cuda.get_device_name(device)
        if bool(pattern.search(gpu)):
            logging.warning(
                f"AMP training on {gpu} GPU may cause "
                f"NaN losses or zero-mAP results, so AMP will be disabled during training."
            )
            return False
    logging.info("AMP training checks successfully.")
    return True


class EarlyStopping:
    def __init__(self, patience: int=50, stop_ratio: float=0.5):
        self.best_metrics = None
        self.patience = patience or float("inf") # epochs to wait after fitness stops improving before stopping
        self.stop_ratio = stop_ratio

        self.possible_stop = False # possible stop may occur next epoch
        self.best_epoch = 0

    def step(self, epoch: int, current_metrics: Dict[str, float], 
             modes: Dict[str, str]):
        logging.info("Early Stopping...")
        if epoch == 0:
            self.best_metrics = {k: float("inf") if v == "min" else -float("inf") for k, v in modes.items()}

        len_metrics = len(current_metrics)
        len_min, len_max = 0., 0.
        if current_metrics is None:
            return False
        
        for k, v in current_metrics.items():
            if modes[k] == "min" and v < self.best_metrics[k]:
                len_min += 1.

            if modes[k] == "max" and v > self.best_metrics[k]:
                len_max += 1.
        
        if (len_min + len_max) / len_metrics >= self.stop_ratio:
            self.best_epoch = epoch
            self.best_metrics = current_metrics
        
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= (self.patience - 1)
        stop = delta >= self.patience

        if stop:
            logging.info(
                f"Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop