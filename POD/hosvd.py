
import numpy as np

def hosvd(X, eps, plot=False):
    """
     Using a user defined projection error as a criteria,
     calculate the minimum projection error onto a subspace of the 
     value function X. 
    
       Parameters:
       -----------
           X(matrix of doubles): Payoff functional; 
           eps (float): (User-defined) projection error;
           plot(bool):  show the projection error on a plot?
    
       Returns:
       --------
       Core: Core tensor (representing the critical mass of X)
       Un: Optimal orthonormal matrix (see paper)
       V: Optimal eigen vectors
       ranks: best rank approximation of X that admits the tolerable decomp
                error
           Basically, it is a projection of X onto one of its subspaces.
       
     Author: Lekan Molu, Dec 3, 2021    
    """
    
    # Get gram matrix
    S = X * X'
    
    # do eigen decomposition of Gram matrix
    V, Lambda = eig(S)
    
    #Find Un
    Sig = np.diag(Lambda, 0)  # collect 0-th diagonal elements
    err_term = (eps^2 * norm(X, 'fro')^2)/ndims(X) 
    
    lambs = np.cumsum(Sig[1:])
    ranks = np.count_nonzero(lambs <= err_term)     
    Un = V[:, :ranks]
    
    Core = X * Un
    return Core, Un,  V, ranks
  