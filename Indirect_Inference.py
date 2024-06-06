# Indirect Inference approach

import numpy as np

def find_closest_rho(phi2phihat, x): # having a rho hat returns the corresponding rho
    return phi2phihat[np.argmin(np.abs(phi2phihat[:, 1] - x)), 0]