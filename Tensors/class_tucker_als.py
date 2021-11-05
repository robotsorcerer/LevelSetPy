__all__ = ["TuckerTensor"]

from .class_tensor import import Tensor

## adhoc functions/classes we'll need to get things rolling
class TuckerTensor(Tensor):
    def __init__(self, core, U):
        """
            Tucker Tensor Class:
                Decomposes a high-order tensor into its core component and
                a set of (usually) unitary matrices associated with every
                mode of the tensor.

            Params
            ------
            core: The core tensor, whose entries shows the interaction among
                  its components
            U:    The factor matrices (typically orthogonal:=principal components
                    in each mode)

            Author: Lekan Molux
            Date: November 2, 2021
        """
        if isinstance(core, Tensor):
            self.Core = core
        else:
            self.Core = Tensor(core, core.shape)

        self.U    = U
