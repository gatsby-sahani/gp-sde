import torch
from torch import nn
from settings import float_type
from quadrature import gauss_legendre
import warnings


class KullbackLeibler(nn.Module):
    def __init__(self, trLen, nLeg=None, dt=None):
        super(KullbackLeibler, self).__init__()
        """
        wquad [N x Q] array of quadrature weights for N trials, Q nodes

        this is to evaluate the integral from 0 to T_n of the KL divergence
        """
        if nLeg is None:
            # don't use quadrature, m, S define grid
            if dt is None:
                raise('need to specify grid spacing if not using quadrature to evaluate integral')
            else:
                self.wwLeg = dt
        else:
            self.nTrials = len(trLen)  # trLen is length R list containing trial lengths
            # Gauss Legendre quadrature nodes and weights
            xxL = torch.zeros(self.nTrials, nLeg).type(float_type)
            wwL = torch.zeros(self.nTrials, nLeg).type(float_type)

            for rr in range(self.nTrials):
                xxL[rr, :], wwL[rr, :] = gauss_legendre(nLeg, a=0, b=trLen[rr])

            self.xxLeg = xxL  # R x n
            self.wwLeg = wwL

    def forward(self, fx, ffx, dfdx, m, S, A, bb):

        # evaluate KL divergence at input locations
        kld = 0.5 * ffx \
            - (fx * bb).sum(-1, keepdim=True) \
            + 0.5 * (bb**2).sum(-1, keepdim=True) \
            - (bb * m.matmul(A.transpose(-1, -2))).sum(-1, keepdim=True) \
            + 0.5 * (A.transpose(-1, -2).matmul(A) * S).sum(-2, keepdim=True).sum(-1, keepdim=True) \
            + 0.5 * ((m.matmul(A.transpose(-1, -2)))**2).sum(-1, keepdim=True) \
            + (dfdx * (A.matmul(S))).sum(-2, keepdim=True).sum(-1, keepdim=True) \
            + (fx * m.matmul(A.transpose(-1, -2))).sum(-1, keepdim=True)

        if torch.any(kld < 0):
            warnings.warn('negative KL-divergence encountered')

        return (kld.squeeze() * self.wwLeg).sum()  # scalar
