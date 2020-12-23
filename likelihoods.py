import torch
from torch import nn
import numpy as np
from utils import bin_spikeTrain, bin_sparseData
from quadrature import gauss_hermite, gauss_legendre
from settings import float_type

# ---------- Point Process Likelihood ------------


class PointProcess(nn.Module):
    def __init__(self, Y, link, trLen, nLeg=50, nHerm=20, dtstep=0.001):
        super(PointProcess, self).__init__()
        """
        Class for PointProcess expected log likelihood
        Y       -- data list with [R][D][Nspikes] trials, neurons, number of spikes
        link    -- non-linearity module
        trLen   -- list with length of each trial
        nLeg    -- number of quadrature nodes to use for Gauss Legendre (to evaluate normalizer)
        nHerm   -- number of Gauss Hermite quadrature nodes to use for evaluating Gaussian expectations of link function (if not analytic)
        """

        self.nTrials = len(trLen)  # number of trials
        self.nOut = len(Y[0])  # number of output dimensions
        self.link = link  # link function object
        self.trLen = trLen  # length of each trial in units of time
        self.dtstep = dtstep  # discretisation for inference
        # Gauss Hermite quadrature nodes and weights
        xxH, wwH = gauss_hermite(nHerm)
        self.xxHerm = xxH
        self.wwHerm = wwH

        # Gauss Legendre quadrature nodes and weights
        xxL = torch.zeros(self.nTrials, nLeg).type(float_type)
        wwL = torch.zeros(self.nTrials, nLeg).type(float_type)

        for rr in range(self.nTrials):
            xxL[rr, :], wwL[rr, :] = gauss_legendre(nLeg, a=0, b=trLen[rr])

        self.xxLeg = xxL  # R x n
        self.wwLeg = wwL

        # arrange data and ensure everything is a pytorch variable
        self.arrange_data(Y)

    def arrange_data(self, Y):

        # ensure data is a pytorch variable

        # concatenate all spikes for each trial into a long list
        Ystack = [torch.tensor(np.concatenate(Y[rr])).type(float_type) for rr in range(self.nTrials)]

        # count number of spikes each neuron fired on each trial
        numSpikes = [[len(Y[rr][nn]) for nn in range(self.nOut)] for rr in range(self.nTrials)]

        # make list for collecting indices
        idxlist = [[] for _ in range(self.nTrials)]

        for rr in range(self.nTrials):
            nn = 0
            for zz in numSpikes[rr]:
                idxlist[rr].append(list(np.tile(nn, (zz,))))
                nn += 1
        # neuron identity for each spike

        neuronIndex = [torch.tensor(sum(bla, []), dtype=torch.long) for bla in idxlist]

        # save stacked version of spikes
        self.Y = Ystack

        # save binned version of spikes
        self.Ybin = [bin_spikeTrain(Y[i], self.trLen[i], self.dtstep) for i in range(self.nTrials)]

        # save indices for which neuron generated which spike
        self.spikeID = neuronIndex

    def closedFormUpdates(self, mu, cov):
        pass

    def expected_loglik(self, mu_sp, cov_sp, mu_qu, cov_qu):
        # mu_qu is R x Tq x D, mu_sp is R x Nsp
        # cov_qu is R x Tq x D,
        # mu_sp is [R] [1 x Tsp x D]
        # mu_sp, cov_sp = self.correct_for_spikeID(mu_sp, cov_sp)

        if self.link.name == 'exponential':
            intval = (self.link(mu_qu + 0.5 * cov_qu) * self.wwLeg.unsqueeze(-1).unsqueeze(-1)).sum(1)  # R x 1 x D
            log_link = mu_sp  # list
        else:
            intval = torch.bmm(self.link(mu_qu + torch.sqrt(2 * cov_qu) * self.xxHerm).mm(self.wwHerm), self.wwLeg)
            log_link = [torch.log(self.link(mu_sp[rr] + torch.sqrt(2 * cov_sp[rr]) * self.xxHerm)).mm(self.wwHerm) for rr in range(len(mu_sp))]  # list

        plik1 = intval.sum()
        plik2 = torch.cat(log_link).sum()

        ell = -plik1 + plik2

        return ell

    def dContinuousdmu(self, mu, cov, idx):
        # get gradient of continuous part of expected log likelihood with respect to mu

        if self.link.name == 'exponential':
            dintval = self.link.derivative(mu + 0.5 * cov)
        else:
            dintval = (self.link.derivative(mu + torch.sqrt(2 * cov) * self.xxHerm).mm(self.wwHerm))

        return -dintval

    def dContinuousdcov(self, mu, cov, idx):
        # get gradient of continuous part of expected log likelihood with respect to cov
        if self.link.name == 'exponential':
            dintval = 0.5 * self.link.derivative(mu + 0.5 * cov)
        else:
            raise('NotImplementedError')
            dintval = self.link.derivative(mu + torch.sqrt(2 * cov) * self.xxHerm).mm(self.wwHerm)

        return -dintval

    def dJumpdmu(self, mu, cov, idx):
        # get gradient of jump part of expected log likelihood with respect to mu
        mask = self.Ybin[idx].unsqueeze(1)

        if self.link.name == 'exponential':
            dlog_link = torch.ones_like(mu)  # T x 1 x D
        else:
            dlog_link = 1. / (self.link(mu + torch.sqrt(2 * cov) * self.xxHerm)).mm(self.wwHerm) * (self.link.derivative(mu + torch.sqrt(2 * cov) * self.xxHerm)).mm(self.wwHerm)

        dlog_link = mask * dlog_link

        # correct for absence of spikes, spike ID in this by zero-masking

        return dlog_link

    def dJumpdcov(self, mu, cov, idx):
        # get gradient of jump part of expected log likelihood with respect to cov

        mask = self.Ybin[idx].unsqueeze(1)

        if self.link.name == 'exponential':
            dlog_link = torch.zeros_like(cov)
        else:
            raise('NotImplementedError')
            dlog_link = torch.log(self.link(mu + torch.sqrt(2 * cov) * self.xxHerm)).mm(self.wwHerm)

        dlog_link = mask * dlog_link

        return dlog_link

    def expected_loglik_gradients(self, mu, cov, idx):
        # gradients with respect to input mean and covariance
        dmu = self.dContinuousdmu(mu, cov, idx)
        dmu_jump = self.dJumpdmu(mu, cov, idx)
        dcov = self.dContinuousdcov(mu, cov, idx)
        dcov_jump = self.dJumpdcov(mu, cov, idx)
        return dmu, dmu_jump, dcov, dcov_jump

    # forward pass to evaluate cost function
    def forward(self, mu_sp, cov_sp, mu_qu, cov_qu):
        ell = self.expected_loglik(mu_sp, cov_sp, mu_qu, cov_qu)
        return ell


# ---------- Poisson Likelihood ------------
class Poisson(nn.Module):
    def __init__(self, Y, link, trLen, nLeg=50, nHerm=20, dtstep=0.001):
        super(Poisson, self).__init__()
        """
        Class for PointProcess expected log likelihood
        Y       -- data list with [R][D][Nspikes] trials, neurons, number of spikes
        link    -- non-linearity module
        trLen   -- list with length of each trial
        nLeg    -- number of quadrature nodes to use for Gauss Legendre (to evaluate normalizer)
        nHerm   -- number of Gauss Hermite quadrature nodes to use for evaluating Gaussian expectations of link function (if not analytic)
        """

        self.nTrials = len(trLen)  # number of trials
        self.nOut = len(Y[0])  # number of output dimensions
        self.link = link  # link function object
        self.trLen = trLen  # length of each trial in units of time
        self.dtstep = dtstep  # discretisation for binning and inference
        # Gauss Hermite quadrature nodes and weights
        xxH, wwH = gauss_hermite(nHerm)
        self.xxHerm = xxH
        self.wwHerm = wwH

        # arrange data and ensure everything is a pytorch variable
        self.arrange_data(Y)

    def closedFormUpdates(self, mu, cov):
        pass

    def arrange_data(self, Y):

        # save binned version of spikes
        self.Ybin = [bin_spikeTrain(Y[i], self.trLen[i], self.dtstep) for i in range(self.nTrials)]

    def expected_loglik(self, mu, cov, idx):
        # mu_qu is R x Tq x D, mu_sp is R x Nsp
        # cov_qu is R x Tq x D,
        # mu_sp is [R] [1 x Tsp x D]
        # mu_sp, cov_sp = self.correct_for_spikeID(mu_sp, cov_sp)
        mask = self.Ybin[idx].unsqueeze(1)

        if self.link.name == 'exponential':
            intval = self.dtstep * self.link(mu + 0.5 * cov)  # R x T x D
            log_link = mu  # list
        else:
            intval = self.link(mu + torch.sqrt(2 * cov) * self.xxHerm).mm(self.wwHerm)
            log_link = [torch.log(self.link(mu + torch.sqrt(2 * cov) * self.xxHerm)).mm(self.wwHerm) for rr in range(len(mu))]  # list

        plik1 = intval.sum()
        plik2 = (mask * log_link).sum()

        ell = -plik1 + plik2
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
        mask = self.Ybin[idx].unsqueeze(1)

        if self.link.name == 'exponential':
            dintval = self.dtstep * self.link.derivative(mu + 0.5 * cov)
            dlog_link = torch.ones_like(mu)  # T x 1 x D
        else:
            dintval = self.dtstep * (self.link.derivative(mu + torch.sqrt(2 * cov) * self.xxHerm).mm(self.wwHerm))
            dlog_link = 1. / (self.link(mu + torch.sqrt(2 * cov) * self.xxHerm)).mm(self.wwHerm) * (self.link.derivative(mu + torch.sqrt(2 * cov) * self.xxHerm)).mm(self.wwHerm)

        djump = - dintval + mask * dlog_link
        return djump

    def dJumpdcov(self, mu, cov, idx):
        # get gradient of jump part of expected log likelihood with respect to cov

        mask = self.Ybin[idx].unsqueeze(1)

        if self.link.name == 'exponential':
            dintval = self.dtstep * 0.5 * self.link.derivative(mu + 0.5 * cov)
            dlog_link = torch.zeros_like(cov)
        else:
            raise('NotImplementedError')
            dintval = self.dtstep * self.link.derivative(mu + torch.sqrt(2 * cov) * self.xxHerm).mm(self.wwHerm)
            dlog_link = torch.log(self.link(mu + torch.sqrt(2 * cov) * self.xxHerm)).mm(self.wwHerm)

        djump = - dintval + mask * dlog_link
        return djump

    def expected_loglik_gradients(self, mu, cov, idx):
        # gradients with respect to input mean and covariance
        dmu = self.dContinuousdmu(mu, cov, idx)
        dmu_jump = self.dJumpdmu(mu, cov, idx)
        dcov = self.dContinuousdcov(mu, cov, idx)
        dcov_jump = self.dJumpdcov(mu, cov, idx)
        return dmu, dmu_jump, dcov, dcov_jump

    # forward pass to evaluate cost function
    def forward(self, mu, cov, idx):
        ell = self.expected_loglik(mu, cov, idx)
        return ell

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
