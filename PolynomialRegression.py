# Polynomial Regression Method

# Hermite Polynomials
def hermite(x, k):
    if k == 0:
        return 1
    elif k == 1:
        return x
    elif k == 2:
        return x**2 - 1
    elif k == 3:
        return x * (x**2 - 3)
    elif k == 4:
        z = x**2
        return z * (z - 6) + 3
    else:
        raise ValueError("Order k > 5")
    
# g(x) function
from scipy.special import logit, expit
import numpy as np

def g(x):
    return logit((x+1)/2)

def inv_g(x):
    return expit(x) * 2 - 1

# f(rho_hat, beta) function
def f(x, beta):
    output = 0
    for i in range(len(beta)):
        output += beta[i] * hermite(x, i)
    output = inv_g(output)
    return output

