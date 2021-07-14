import pandas as pd
import numpy as np
import scipy.stats as sp
import itertools
import scipy.optimize as opt
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


def fit3D(x_wire, x_drift, x_label, y):
    if len(x_wire) < 3 or len(x_wire) != len(x_drift) or len(x_drift) != len(x_label) or len(x_label) != len(y): return 0., 0., 999., 999.
    y_wire, y_drift, y_label, x = x_wire.copy(), x_drift.copy(), x_label.copy(), y.copy()
    y_label[y_label == 1] = -1
    y_label[y_label == 2] = +1
    y = y_wire + y_label * y_drift
    n, xsum, ysum = len(x), x.sum(), y.sum()
    ovb = (n * (x*y).sum() - xsum * ysum) / (n * (x**2).sum() - (xsum)**2 )
    ova = (ysum - ovb * xsum) / n
    if ovb == 0.: ovb = 1.e9
    f = ovb * x + ova
    chi2 = ((f - y)**2 ).sum() / (n - 1)
    
    # Nested function
    def chiSquare3D(args):
        _ovb, _ova, _dy = args
        _f = _ovb * x + _ova
        _y = y_wire + y_label * (y_drift + _dy)
        chi2 = ( (_f - _y)**2 ).sum() / (n - 1)
        return chi2
    
    x0 = np.array([ovb, ova, 0.])
    bnds = None #((None, None), (None, None), (0, 21.))
    result =  opt.minimize(chiSquare3D, x0, bounds=bnds)
    fovb, fova, fdy = result['x']
    fchi2 = result['fun']
    
    
    #_chi2 = (((fa*x + fb) - (y + l*fdx))**2 ).sum() / (n - 1) OK
    
    #print("chi2: ", chi2, fchi2)
    #print("a   : ", ova, fova)
    #print("b   : ", ovb, fovb)
    #print("dx  : ", 0., fdy)
    #print(result)
    
    # Re-rotate back to normal reference system and prepare output
    b, a = 1./ovb, -ova/ovb
    
    if not result['status'] == 0: return 0., 0., 999., 999.
    return b, a, fchi2, fdy

### -------------------------------------------

