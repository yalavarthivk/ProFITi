"""Base model interface for ProFITi components."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
from torch import Tensor


class BaseFlowModel(ABC):
    """Abstract base class for flow models."""

    @abstractmethod
    def forward_flow(self, y: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply forward flow transformation."""
        pass

    @abstractmethod
    def inverse_flow(self, z: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply inverse flow transformation."""
        pass
