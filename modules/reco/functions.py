import pandas as pd
import numpy as np
import scipy.stats as sp
import itertools
#import numba as nb



# Swap x and z, as the dominating uncertainty is on x

def fitPoly(x, y): # real   7m42.267s user    5m16.005s sys 0m7.780s
    if len(x) < 3: return 0., 0., 999.
    pfit = np.polyfit(y, x, 1, full=False)
    if pfit[0] == 0.: return 0., 0., 999.
    f = pfit[0] * y + pfit[1]
    chi2 = ((f - x)**2 ).sum() / (len(x) - 1)
    m, q = 1./pfit[0], -pfit[1]/pfit[0]
    return m, q, chi2

def fitPolyFull(x, y): # real   6m22.913s user    4m45.118s sys 0m3.578s
    if len(x) < 3: return 0., 0., 999.
    pfit, residuals, rank, singular_values, rcond = np.polyfit(y, x, 1, full=True)
    if len(residuals) <= 0 or pfit[0] == 0.: return 0., 0., 999.
    return 1./pfit[0], -pfit[1]/pfit[0], residuals[0]/(len(y) - 1)

def fitLstsq(x, y): # real  5m22.888s user    4m19.011s sys 0m3.571s
    if len(x) < 3: return 0., 0., 999.
    pfit, residuals, _, _ = np.linalg.lstsq(np.vstack([y, np.ones(len(y))]).T, x, rcond=None)
    if pfit[0] == 0.: return 0., 0., 999.
    return 1./pfit[0], -pfit[1]/pfit[0], residuals[0]/(len(x) - 1)


def fitLinregress(x, y): # real    6m18.321s user    5m13.408s sys 0m3.793s
    if len(x) < 3: return 0., 0., 999.
    ovm, ovq, _, _, _ = sp.linregress(y, x)
    if ovm == 0.: return 0., 0., 999.
    f = ovm * y + ovq
    chi2 = ((f - x)**2 ).sum() / (len(x) - 1)
    m, q = 1./ovm, -ovq/ovm
    return m, q, chi2

#@nb.jit(nopython=True, fastmath = True, parallel = True)
def fitFast(x, y):
    if len(x) < 3 or len(x) != len(y): return 0., 0., 999.
    y, x = x, y
    n, xsum, ysum = len(x), x.sum(), y.sum()
    ovb = (n * (x*y).sum() - xsum * ysum) / (n * (x**2).sum() - (xsum)**2 )
    ova = (ysum - ovb * xsum) / n
    if ovb == 0.: return 0., 0., 999.
    f = ovb * x + ova
    chi2 = ((f - y)**2 ).sum() / (n - 1)
    b, a = 1./ovb, -ova/ovb
    return b, a, chi2

### -------------------------------------------

