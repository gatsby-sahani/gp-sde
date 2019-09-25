import torch
from torch import nn
from settings import float_type
from utils import batch_make_diag, unsqueeze_as, batch_vec_to_diag, batch_inverse, batch_diag, apply_along_axis, logdet
from settings import jitter_level, mean_init, var_init
import warnings
import pdb

# --------------- base class for output mappings ---------------


class OutputMapping(nn.Module):
    def __init__(self):
        super(OutputMapping, self).__init__()
    """
    Base class for output mapping
    """

    def refresh_stored_values(self):
        pass

    def log_prior_distribution(self):
        # for cost function computation
        return torch.zeros(1, 1).type(float_type)

    def dOutdm(self, m, S):
        raise NotImplementedError()

    def dOutdS(self, m, S):
        raise NotImplementedError()

    def OutMean(self, m, S, outIdx=None):
        raise NotImplementedError()

    def OutCov(self, m, S, outIdx=None):
        raise NotImplementedError()

    def forward(self, m, S, outIdx=None):
        # m is T x 1 x K (or R x T x 1 x K -- batch extends first dim)
        # S is T x K x K (or R x T x 1 x K)
        # outIdx says which output dimensions to return per time index, by default all are returned
        mu = self.OutMean(m, S, outIdx)
        cov = self.OutCov(m, S, outIdx)
        prior = self.log_prior_distribution()
        return mu, cov, prior


class AffineMapping(OutputMapping):
    def __init__(self, C0, d0, useClosedForm=False):
        super(AffineMapping, self).__init__()
        self.useClosedForm = useClosedForm

        if self.useClosedForm:
            self.Subspace = C0.type(float_type)  # K x D
            self.Offset = d0.type(float_type)  # 1 x D
        else:
            self.Subspace = nn.Parameter(C0.type(float_type))  # K x D
            self.Offset = nn.Parameter(d0.type(float_type))  # 1 x D
    """
    Affine output mapping: g(x) = Cx + d
    C0 is the K x D initial value of the Subspace mapping
    d0 is the 1 x D initial value o fthe constant offset
    """

    def dOutdm(self, m, S):
        with torch.no_grad():
            dmudm = self.Subspace.permute(1, 0)  # D x K
            dcovdm = torch.zeros_like(dmudm)
        # output are 1 x D x K
        return dmudm.unsqueeze(0), dcovdm.unsqueeze(0)

    def dOutdS(self, m, S):
        with torch.no_grad():
            dcovdS = self.Subspace.unsqueeze(-1).permute(1, 2, 0) * self.Subspace.unsqueeze(-1).permute(1, 0, 2)  # D x K x K
            dcovdS = dcovdS + dcovdS.transpose(-1, -2) - batch_make_diag(dcovdS)  # account for symmetry in S
            dmudS = torch.zeros_like(dcovdS)
        # outputs are 1 x D x K x K
        return dmudS.unsqueeze(0), dcovdS.unsqueeze(0)

    def orthogonaliseLatents(self, m, S):
        V, D, U = torch.svd(self.Subspace)
        Corth = U[:]
        Dd = torch.diag(D)
        m_rot = m.matmul(V).matmul(Dd)
        S_rot = Dd.matmul(V.permute(1, 0)).matmul(S).matmul(V).matmul(Dd)
        return m_rot, S_rot, Corth

    def OutMean(self, m, S, outIdx=None):
        if outIdx is None:
            mu = torch.matmul(m, self.Subspace) + self.Offset  # T x 1 x D
        else:  # return desired output dimension at given time point
            assert(outIdx.size()[0] == m.size(0))  # one index per time point
            mu = (m * self.Subspace.index_select(1, outIdx).unsqueeze(-1).permute(1, 2, 0)).sum(-1, keepdim=True) \
                + self.Offset.index_select(1, outIdx).unsqueeze(-1).permute(1, 2, 0)  # T x 1 x 1
        return mu

    def OutCov(self, m, S, outIdx=None):
        if outIdx is None:
            cov = (self.Subspace * S.matmul(self.Subspace)).sum(-2, keepdim=True)  # T x 1 x D
        else:  # return desired output dimension at given time point
            assert(outIdx.size()[0] == m.size(0))  # one index per time point
            cov = (self.Subspace.index_select(1, outIdx).unsqueeze(-1).permute(1, 0, 2) *
                   S.matmul(self.Subspace.index_select(1, outIdx).unsqueeze(-1).permute(1, 0, 2))).sum(-2, keepdim=True)  # T x 1 x 1
        return cov

    def closedFormUpdates(self, like, m, S, m_qu, S_qu):
        if self.useClosedForm:
            if like.__class__.__name__ is 'Gaussian':
                with torch.no_grad():
                    K, D = self.Subspace.size()

                    XX = torch.zeros(K, K)
                    XY = torch.zeros(K, D)
                    for idx in range(like.nTrials):
                        # (Y - d)m'
                        XY += ((like.Y[idx].unsqueeze(1) - self.Offset) * m[idx].transpose(-2, -1)).sum(0)  # K x D
                        # S + mm'
                        XX += (S[idx] + m[idx].matmul(m[idx].transpose(-1, -2))).sum(0)  # K x K

                    Cnew, _ = torch.solve(XY, XX)

                    Ydiff = torch.zeros(1, D)
                    Ntot = 0
                    for idx in range(like.nTrials):
                        Ydiff += (like.Y[idx].unsqueeze(1) - torch.matmul(m[idx], Cnew)).sum(0)
                        Ntot += like.nTimes[idx]

                    dnew = (Ydiff).div(Ntot)

                    self.Subspace = Cnew  # K x D
                    self.Offset = dnew.view(1, -1)  # 1 x D
            else:
                raise NotImplementedError()

        else:
            pass
