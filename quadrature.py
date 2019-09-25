import numpy as np
import torch
from settings import float_type


def gauss_legendre(n, a=-1., b=1.):
    """function to compute weights and abscissas for Gauss-Legendre quadrature """
    h = b - a
    x, w = np.polynomial.legendre.leggauss(n)

    # adjust integration limits
    if a != -1. or b != 1.:
        x = (x + 1.) * (h / 2.) + a
        w = (h / 2.) * w

    xt = torch.tensor(x).type(float_type)
    wt = torch.tensor(w).type(float_type)
    return xt, wt


def gauss_hermite(n):
    """function to compute weights and abscissas for Gauss-Hermite quadrature"""
    x, w = np.polynomial.hermite.hermgauss(n)
    xt = torch.tensor(x).type(float_type)
    wt = torch.tensor(w).type(float_type)

    return xt, wt
