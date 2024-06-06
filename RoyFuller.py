# Roy-Fuller median-unbiased estimator
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import numpy as np

def RF_BiasCorrection(sequence, tau_half):
    
    ols_rho = AutoReg(sequence, 1).fit().params[1]
    ols_se = AutoReg(sequence, 1).fit().bse[1]                  
    
    tau = (ols_rho - 1) / ols_se

    if tau <= -np.sqrt(2 * len(sequence)):
        C = 0
    elif tau > -np.sqrt(2 * len(sequence)) and tau <= -5:
        C = tau / len(sequence) - 2 / tau
    elif tau > -5 and tau <= tau_half:
        k2 = (2 - tau_half**2 / len(sequence)) / ((1 + 1/len(sequence)) * tau_half * (tau_half + 5))
        C = tau / len(sequence) - 2 / (tau + k2 * (tau + 5) )
    else:
        C = - tau_half + 0.1111 * (tau - tau_half)

    tau_minus = (ols_rho + 1) / ols_se
    
    if tau_minus >= np.sqrt(len(sequence) / 3):
        C_minus = 0
    elif tau_minus < np.sqrt(len(sequence) / 3) and tau_minus >= 5:
        C_minus = 3 * tau_minus / len(sequence) - 1 / tau_minus
    else:
        C_minus = 0.0467 + 0.0477 * (tau_minus - 5)

    bias = (C + C_minus) * ols_se
    return bias

  
