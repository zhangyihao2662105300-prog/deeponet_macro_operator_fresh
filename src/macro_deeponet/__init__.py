"""Fresh DeepONet macro-element strain operator package."""

from .autograd import strain_jacobian_wrt_q
from .models import MacroDeepONet

__all__ = ["MacroDeepONet", "strain_jacobian_wrt_q"]

