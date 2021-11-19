__all__ = ['Tensor']

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Tensor Algebra"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Finished"

import numpy as np
import numpy as np

class Tensor():
    def __init__(self, data=None, shape=(), rindices=None, cindices=None):
        """
            Tensor class: This class wraps a numpy or cupy array
            into a Tensor class

            Parameters:
                array: Array that contains the data owned by the tensor.

                shape: Shape of the tensor, or shape to cast array to within the tensor.
        """

        #assert len(shape)>=1, 'shape of Tensor cannot be null.'
        if len(shape)<1:
            # infer array shape from data
            shape = data.shape

        assert np.any(data), 'Tensor cannot hold empty array.'

        if np.any(data):
            if isinstance(data, np.ndarray):
                self.type = 'numpy'
            elif isinstance(data, np.ndarray):
                self.type = 'cupy'
            elif isinstance(data, list):
                raise ValueError("Only supports Numpy and CuPy Ndarrays at this time.")

        self.data = data

        if ((len(shape)>0) and (self.data.shape!=shape)):
            self.data = self.data.reshape(shape)
            
        # ad-hoc attributes for tensor matricization
        
        self.rindices  = rindices
        self.cindices  = cindices

    @property
    def ndim(self):
        return self.data.ndim

    
    @property
    def tsize(self):
        return self.shape
            
    @property
    def shape(self):
        return list(self.data.shape)

    def __dtype__(self):
        return f"Tensor"


class KruskalTensor():
    def __init__(self, T, U):
        """
            Tensor class: This class wraps a numpy or cupy array
            into a Tensor class

            Parameters:
                array: Array that contains the data owned by the tensor.

                shape: Shape of the tensor, or shape to cast array to within the tensor.
        """
        if not isinstance(T, Tensor):
            if isinstance(T, np.ndarray):
                self.type = 'numpy'
            elif isinstance(T, np.ndarray):
                self.type = 'cupy'
            elif isinstance(T, list):
                raise ValueError("Only supports Numpy and CuPy Ndarrays at this time.")

            self.T = Tensor(T, T.shape)
        else:
            self.T = T

        self.T.U = U

    def __dtype__(self):
        return f"KTensor"

    @property
    def ndim(self):
        return self.T.ndim
