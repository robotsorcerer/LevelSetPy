__all__ = ["dubins_absolute", "dubins_asym", "dubins_sym"]

# Dubins car dynamics in absolute coordinates

import numpy as np
from LevelSetPy.Utilities import *

def dubins_absolute(obj, p1, capture_radius=.1, avoid_radius=.5):
    """
        Dubins car dynamics in absolute coordinates.
        This is useful for multiple pursuers and
        evaders. 
        
        This function resolves the separate dynamics on
        different grids in the hopes that they can be respectively
        stitched together.

        The velocities are assumed equal similar to Merz's 
        1972 constrution
        
    
        Parameters
        ==========
        obj: Bundle type with the following fields:            
            v: Linear velocity of each car
                ve: Linear velocity of evader
                vp: Linear velocity of pursuer         
            w: Angular velocity of each car
                ve: Angular velocity of evader
                vp: Angular velocity of pursuer 

        p1: agent 1 (pursuer) as a bundle
            .grid
            .center

        Returns
        =======
        xdot: A bundle struct consisting of:
            dyn: The dynamics of the vehicle

        Lekan Molux, Nov. 21, 2021
    """
    assert isfield(p1, 'center'), 'player I must have a center '\
                                    'defined for its capture equation.'

    # get the states for player pairs
    x1    = p1.grid.xs
    xdot  = cell(3)

    xdot[0] = obj.vp*np.cos(x1[2])
    xdot[1] = obj.vp*np.sin(x1[2])
    xdot[2] = obj.we*np.ones_like(x1[-1])

    return xdot

def dubins_sym(obj, p1, p2, mode='avoid', 
                capture_radius=.1, avoid_radius=.5):
    """
        Dubins car dynamics in absolute coordinates.
        This is useful for multiple pursuers and
        evaders. 
        
        This function resolves the separate dynamics on
        different grids in the hopes that they can be respectively
        stitched together.

        The velocities are assumed equal similar to Merz's 
        1972 constrution
        
    
        Parameters
        ==========
        obj: Bundle type with the following fields:            
            v: Linear velocity of each car
                ve: Linear velocity of evader
                vp: Linear velocity of pursuer         
            w: Angular velocity of each car
                ve: Angular velocity of evader
                vp: Angular velocity of pursuer 

        p1: agent 1 (pursuer) as a bundle
            .grid
            .center
        p2: agent 2 (evader)

        mode: whether the joint dynamics for these agent interactions results
        in a 'capture' or 'avoid'.

        Returns
        =======
        xdot: A bundle struct consisting of:
            evader_dyn: The dynamics of the evader pair;
            pursuer_dyn: The dynamics of the pursuer.

        Lekan Molux, Nov. 21, 2021
    """
    assert isfield(p1, 'center'), 'player I must have a center '\
                                    'defined for its capture equation.'
    assert isfield(p2, 'center'), 'player II must have a center '\
                                    'defined for its capture equation.'
    assert isfield(p1, 'grid'), 'player I must have its grid info'
    assert isfield(p2, 'grid'), 'player II must have its grid info'

    # get the states for player pairs
    x1 = p1.grid.xs
    x2 = p2.grid.xs

    p1_dyn  = cell(3)
    p2_dyn = cell(3)

    p1_dyn[0] = obj.vp*np.cos(x1[2])
    p1_dyn[1] = obj.vp*np.sin(x1[2])
    p1_dyn[2] = obj.we

    p2_dyn[0] = obj.ve*np.cos(x2[2])
    p2_dyn[1] = obj.ve*np.sin(x2[2])
    p2_dyn[2] = obj.wp

    assert p1.grid.shape==p2.grid.shape, "Both grids must have equatl shape."
    
    target_set = np.zeros(p1.grid.shape)

    if isfield(p1, 'center') and numel(p1.center==1):
        p1.center = p1.center*np.ones((p1.grid.dim,1), dtype=np.float64)
    if isfield(p2, 'center') and numel(p2.center==1):
        p2.center = p2.center*np.ones((p2.grid.dim,1), dtype=np.float64)
        
    #if mode=='capture':
    target_set = np.zeros(p1.grid.shape)
    for i in range(p1.grid.dim):
        if(i != p1.grid.axis_align):
            target_set += ((p1.grid.xs[i]-p1.center[i]) - 
                            (p2.grid.xs[i]-p2.center[i]))**2
    target_set = np.sqrt(target_set) - capture_radius
    xdot = Bundle(dict(p1=p1_dyn, p2=p2_dyn, targ=target_set))

    return xdot

def dubins_asym(obj, x, w):
    """
        Dubins car dynamics in absolute coordinates.
        This is useful for multiple pursuers and
        evaders. This function resolves the pursuer-evadeerr dynamics on
        same grids.

        Parameters
        ==========
        x: states containing vehicle  1 and 2
            respectively.
        w: Bundle class w/attributes we, wp

        Returns
        =======
        xdot: A bundle struct consisting of:
            evader_dyn: The dynamics of the evader pair;
            pursuer_dyn: The dynamics of the pursuer.

        Lekan Molux, Nov. 21, 2021
    """
    evader_dyn  = cell(6)
    pursuer_dyn = cell(6)

    evader_dyn[0] = obj.v*np.cos(x[2])
    evader_dyn[1] = obj.v*np.sin(x[2])
    evader_dyn[2] = w.we

    pursuer_dyn[2] = obj.v*np.cos(x[5])
    pursuer_dyn[3] = obj.v*np.sin(x[5])
    pursuer_dyn[4] = w.wp

    xdot = Bundle(dict(evader=evader_dyn, pursuer=pursuer_dyn))

    return xdot