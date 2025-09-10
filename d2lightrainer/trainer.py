"""Single GPU basic trainer class"""
import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Literal, Dict, Tuple, Union


class BaseTrainer:
    """
    - Train (Need Loss, Train Loader, Optimizer, Scheduler).
    - Validate (Need Loss, Validate Loader).
    
    BaseTrainer have some instance attributes, we can use `super().__init__()` to inherit them.
    """
    def __init__(self, model: Union[nn.Module, Tuple[nn.Module]], dataset: Dataset):
        self.model = model
        self.dataset = dataset
        
    def get_loader(self, dataset: Dataset,
                   mode: Literal["train", "notrain"]="train"):
        """
        Creates a data loader.

        Constructs a DataLoader instance based on the specified mode (training or validation).
        In training mode, data shuffling is enabled; in validation mode, data is loaded sequentially.

        Args:
            dataset (Dataset): The dataset to be loaded.
            mode (Literal["train", "val"]): The data loading mode.
                - "train": Enables shuffling for training.
                - "val": Disables shuffling for validation.

        Returns:
            DataLoader: A configured DataLoader instance.

        Note:
            - Batch size and number of worker processes are retrieved from the configuration.
            - Data is shuffled during training to improve generalization.
            - Data order is preserved during validation to ensure reproducibility.
        """
        raise NotImplementedError("Not Implemented.")
    
    def get_loss(self, model_out, batch):

        raise NotImplementedError("Not Implemented.")
    
    def get_model_out(self, batch):

        raise NotImplementedError("Not Implemented.")

    def train(self):
        """
        Main training routine to be implemented by subclasses.

        This method must be overridden in derived classes and should include the full
        training loop logic, such as forward pass, loss computation, backward pass, 
        and parameter updates. A typical implementation should cover:
        - Core training loop logic
        - Periodic validation and metric evaluation
        - Model checkpointing
        - Early stopping logic
        - Learning rate scheduling

        Raises:
            NotImplementedError: This method must be implemented in a subclass.

        Note:
            When implementing this method, `setup_train()` should be called first to
            initialize the optimizer, scheduler, and other training components.
        """
        raise NotImplementedError("Not Implemented.")

    @torch.inference_mode()
    def validate(self) -> Tuple[Dict[str, float], Dict[str, str]] | torch.Tensor:
        """
        Performs model validation.
        """
        raise NotImplementedError("Not Implemented.")