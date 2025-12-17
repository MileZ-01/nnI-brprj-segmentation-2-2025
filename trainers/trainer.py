# Imports
from typing import Union, Tuple, List
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class trainer(nnUNetTrainer):
    """
    Custom nnU-Net trainer class that wraps the standard nnUNetTrainer.

    This class allows overriding of the network architecture construction,
    for example, to add extra input channels or modify network settings
    while keeping all nnU-Net training, loss, optimizer, and checkpointing functionality.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the trainer.

        Calls the parent nnUNetTrainer constructor to ensure all
        internal properties (datasets, loss, optimizer, etc.) are properly initialized.

        Args:
            *args: Positional arguments passed to nnUNetTrainer.
            **kwargs: Keyword arguments passed to nnUNetTrainer.
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True
    ) -> nn.Module:
        """
        Build the neural network architecture using the standard nnUNetTrainer method.

        This method can be overridden to customize the number of input or output channels,
        modify the architecture, or integrate interactive channels.

        Args:
            architecture_class_name (str): Name of the network class to instantiate.
            arch_init_kwargs (dict): Keyword arguments for the network constructor.
            arch_init_kwargs_req_import (list or tuple of str): Required modules for dynamic imports.
            num_input_channels (int): Number of input channels (e.g., image + extra channels).
            num_output_channels (int): Number of output channels (e.g., segmentation classes).
            enable_deep_supervision (bool, optional): Whether to enable deep supervision. Defaults to True.

        Returns:
            nn.Module: The constructed PyTorch neural network module.
        """
        print(f"Building network architecture...")
        print(f"  Input channels: {num_input_channels}")
        print(f"  Output channels: {num_output_channels}")

        network = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=enable_deep_supervision
        )

        print(f"  Network built with {num_input_channels} total input channels")
        return network