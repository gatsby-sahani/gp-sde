import torch
from torch import nn
import numpy as np
from utils import bin_spikeTrain, bin_sparseData
from quadrature import gauss_hermite, gauss_legendre
from settings import float_type


# ---------- Gaussian Likelihood ------------
class Gaussian(nn.Module):
    def __init__(self, Y, tObs, trLen, R0, dtstep=0.001, useClosedForm=True):
        super(Gaussian, self).__init__()
        """
        Class for PointProcess expected log likelihood
        Y       -- data list [R][T x D]
        trLen   -- list with length of each trial
        tObs    -- time points where an observations was made
        """
        self.nTrials = len(trLen)  # number of trials
        self.nOut = len(Y[0])  # number of output dimensions
        self.trLen = trLen  # length of each trial in units of time
        self.dtstep = dtstep  # discretisation used for solving ODEs in inference
        self.useClosedForm = useClosedForm
        if self.useClosedForm:
            self.outputVariance = R0.type(float_type).view(1, -1)  # 1 x D
        else:
            self.outputVariance = nn.Parameter(R0.type(float_type).view(1, -1))  # initial value for variance

        self.nTimes = [len(tObs[i]) for i in range(self.nTrials)]
        # arrange data and ensure everything is a pytorch variable
        self.arrange_data(Y, tObs)

    def arrange_data(self, Y, tObs):
        self.tObs = [torch.tensor(tObs[i]).type(float_type) for i in range(self.nTrials)]  # list of time points for each trial where observations were made for given trial
        self.Y = [torch.tensor(Y[i]).type(float_type) for i in range(self.nTrials)]  # T x D
        self.Ybin = [bin_sparseData(self.Y[i], self.tObs[i], self.trLen[i], self.dtstep) for i in range(self.nTrials)]  # places observations on grid for jump conditions
        self.Ysq = [self.Y[i]**2 for i in range(self.nTrials)]  # each entry is T x D

    def expected_loglik(self, mu, cov, idx):
        # expected log likelihood for trial idx
        # mu is T x 1 x D
        # cov is T x 1 x D
        ell = - 0.5 * (self.Ysq[idx].unsqueeze(1) - 2 * self.Y[idx].unsqueeze(1) * mu + mu**2 + cov).div(self.outputVariance.view(1, 1, -1)).sum() - \
            0.5 * self.nTimes[idx] * torch.log(self.outputVariance).sum()
        return ell

    def dContinuousdmu(self, mu, cov, idx):
        # get gradient of continuous part of expected log likelihood with respect to mu
        dcont = torch.zeros_like(mu)
        return dcont

    def dContinuousdcov(self, mu, cov, idx):
        # get gradient of continuous part of expected log likelihood with respect to cov
        dcont = torch.zeros_like(cov)
        return dcont

    def dJumpdmu(self, mu, cov, idx):
        # get gradient of jump part of expected log likelihood with respect to mu
        mask = (self.Ybin[idx].unsqueeze(1) != 0).type(float_type)
        djump = (self.Ybin[idx].unsqueeze(1) - mu).div(self.outputVariance.view(1, 1, -1))
        return djump * mask

    def dJumpdcov(self, mu, cov, idx):
        # get gradient of jump part of expected log likelihood with respect to cov
        mask = (self.Ybin[idx].unsqueeze(1) != 0).type(float_type)
        djump = -0.5 * torch.ones_like(cov).div(self.outputVariance.view(1, 1, -1))
        return djump * mask

    def expected_loglik_gradients(self, mu, cov, idx):
        # gradients with respect to input mean and covariance
        dmu = self.dContinuousdmu(mu, cov, idx)
        dmu_jump = self.dJumpdmu(mu, cov, idx)
        dcov = self.dContinuousdcov(mu, cov, idx)
        dcov_jump = self.dJumpdcov(mu, cov, idx)
        return dmu, dmu_jump, dcov, dcov_jump

    def closedFormUpdates(self, mu, cov):
        # perform update for the output variance
        if self.useClosedForm:
            with torch.no_grad():
                new_var = torch.zeros_like(self.outputVariance)
                Ntot = 0
                for idx in range(self.nTrials):
                    new_var += (self.Ysq[idx].unsqueeze(1) - 2 * self.Y[idx].unsqueeze(1) * mu[idx] + mu[idx]**2 + cov[idx]).sum(0).view(1, -1)
                    Ntot += self.nTimes[idx]
                # pdb.set_trace()
                self.outputVariance = new_var.div(Ntot)

    # forward pass to evaluate cost function
    def forward(self, mu, cov, idx):
        ell = self.expected_loglik(mu, cov, idx)
        return ell
