# Jackknife approach
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def JK(sequence):
    n = len(sequence)
    midpoint = n // 2
    rho = AutoReg(sequence, 1).fit().params[1]
    rho1 = AutoReg(sequence[:midpoint], 1).fit().params[1]
    rho2 = AutoReg(sequence[midpoint:], 1).fit().params[1]
    
    return 2 * rho - (rho1 + rho2) / 2