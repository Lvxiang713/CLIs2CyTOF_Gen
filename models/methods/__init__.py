from .base_method import BaseGenerativeMethod
try:
    from .flow_matching import FlowMatchingMethod
except Exception:
    FlowMatchingMethod = None
from .ddpm import DDPMMethod
from .vae import VAEMethod
from .gan import GANMethod

__all__ = [
    "BaseGenerativeMethod",
    "FlowMatchingMethod",
    "DDPMMethod",
    "VAEMethod",
    "GANMethod",
]
