__all__ = ['Tensor']

import cupy as cp
import numpy as np

class Tensor():
    def __init__(self, array=None, shape=()):
        """
            Tensor class: This class wraps a numpy or cupy array
            into a Tensor class
            
            Parameters:
                array: Array that contains the data owned by the tensor.
                
                shape: Shape of the tensor, or shape to cast array to within the tensor.
        """
        
        assert len(shape)>=1, 'shape of Tensor cannot be null.'
        
        assert np.any(array), 'Tensor cannot hold empty array.'
        
        if np.any(array):
            if isinstance(array, np.ndarray):
                self.type = 'numpy'
            elif isinstance(array, cp.ndarray):
                self.type = 'cupy'
            elif isinstance(array, list):
                raise ValueError("Only supports Numpy and CuPy Ndarrays at this time.")

        self.data = array

        if ((len(shape)>0) and (self.data.shape!=shape)):
            self.data = self.data.reshape(shape)
            
    @property
    def shape(self):
        return self.data.shape
