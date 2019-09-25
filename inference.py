import torch
from utils import linInterp, batch_make_diag
from settings import float_type, var_init


class GaussMarkovLagrange(object):
    def __init__(self, nLatent, trLen, dtstep=0.001, learningRate=0.9, parallel=0):
        """
        Variational approximation to a non-linear SDE by a time-varying Gauss-Markov Process of the form
        dx(t) = - A(t) dt + b(t) dW(t)
        """
        self.nTrials = len(trLen)  # number of trials
        self.trLen = trLen  # trial lengths
        self.nLatent = nLatent  # number of latents / latent dimensionality
        self.t_grid = [torch.linspace(dtstep, trLen[i], int(trLen[i] / dtstep)).type(float_type) for i in range(self.nTrials)]  # discretized time grid for inference
        self.Psi = [torch.zeros(int(trLen[i] / dtstep), self.nLatent, self.nLatent).type(float_type) for i in range(self.nTrials)]  # T x K x K
        self.eta = [torch.zeros(int(trLen[i] / dtstep), 1, self.nLatent).type(float_type) for i in range(self.nTrials)]  # T x 1 x K
        self.dtstep = dtstep  # step size for inference discretization
        self.learningRate = learningRate  # learning rate for Agrid, bgrid updates
        self.convergenceTol = 1e-04
        self.initialiseVariational()

    def initialiseVariational(self):
        # initialise stored grid
        # A is [R] [T x K x K]
        # b is [R] [T x 1 x K]

        self.A_grid = [(1. / var_init) * torch.eye(self.nLatent, self.nLatent).unsqueeze(0).repeat(int(self.trLen[i] / self.dtstep), 1, 1).type(float_type) for i in range(self.nTrials)]

        self.b_grid = [torch.zeros(int(self.trLen[i] / self.dtstep), 1, self.nLatent).type(float_type) for i in range(self.nTrials)]

    def solveBackward_LagrangeMultipliers(self, model, m, S, dLdm, dLdm_jump, dLdS, dLdS_jump, idx):
        # solve Lagrange multipliers and incorporate jump conditions due to observations
        # timepts are the time points corresponding to the discretisation

        # get cost function gradients at given discretization
        dEdm, dEdS = self.get_KullbackLeibler_grad(model, m, S, idx)

        # set final condition
        Psi_new = torch.zeros_like(self.Psi[idx]).type(float_type)  # T x D x D
        eta_new = torch.zeros_like(self.eta[idx]).type(float_type)  # T x 1 x D

        # symmetry mask which halves gradient contributions of off-diagonal terms (this is bc we are taking derivatives wrt symmetric S, not general S)
        # mask contains ones on the diagonal, ones off the diagonal.
        symmetryMask = 0.5 * torch.ones(self.nLatent, self.nLatent).type(float_type) + 0.5 * torch.eye(self.nLatent, self.nLatent).type(float_type)

        # integrate backwards using Forward Euler method with reversed time grid
        T = Psi_new.shape[0]
        for tt in range(T - 1, 0, -1):  # goes from 1 to T-1 for indexing
            Psi_new[tt - 1, :, :] = Psi_new[tt, :, :] - self.dtstep * (torch.matmul(self.A_grid[idx][tt, :, :].transpose(-2, -1), Psi_new[tt, :, :]) +
                                                                       torch.matmul(Psi_new[tt, :, :], self.A_grid[idx][tt, :, :]) -
                                                                       dEdS[tt, :, :] * symmetryMask + dLdS[tt, :, :] * symmetryMask) - dLdS_jump[tt - 1, :, :] * symmetryMask

            eta_new[tt - 1, :, :] = eta_new[tt, :, :] - self.dtstep * (eta_new[tt, :, :].matmul(self.A_grid[idx][tt, :, :]) - dEdm[tt, :, :] +
                                                                       dLdm[tt, :, :]) - dLdm_jump[tt - 1, :, :]
        Psi_new = 0.5 * (Psi_new + Psi_new.transpose(-2, -1))  # enforce symmetry for stability

        # update stored values at grid points
        self.Psi[idx] = Psi_new[:]
        self.eta[idx] = eta_new[:]

    def get_KullbackLeibler_grad(self, model, m, S, idx):
        # gradients for Kullback leibler divergence
        # gradients wrt m are (R x 1 x 1) x (1 x K)
        with torch.no_grad():
            dEdm_grid = 0.5 * model.transfunc.dffdm(m, S) \
                - (model.transfunc.dfdm(m, S) * (self.b_grid[idx].unsqueeze(-1).unsqueeze(-1))).sum(2, keepdim=True) \
                + m.matmul(self.A_grid[idx].transpose(-1, -2)).matmul(self.A_grid[idx]).unsqueeze(1).unsqueeze(1) \
                - self.b_grid[idx].matmul(self.A_grid[idx]).unsqueeze(1).unsqueeze(1) \
                + model.transfunc.f(m, S).matmul(self.A_grid[idx]).unsqueeze(1).unsqueeze(1) \
                + (model.transfunc.dfdm(m, S) * m.matmul(self.A_grid[idx].transpose(-1, -2)).unsqueeze(-1).unsqueeze(-1)).sum(2, keepdim=True) \
                + (model.transfunc.ddfdxdm(m, S) * self.A_grid[idx].matmul(S).unsqueeze(-1).unsqueeze(-1)).sum(1, keepdim=True).sum(2, keepdim=True)

            # gradients wrt S are  (R x 1 x 1) x (K x K)
            # part of gradient that is already symmetrised (since grads come from transition function, which expects proper gradients)
            dEdS_grid_sym = 0.5 * model.transfunc.dffdS(m, S) \
                - (model.transfunc.dfdS(m, S) * (self.b_grid[idx].unsqueeze(-1).unsqueeze(-1))).sum(2, keepdim=True) \
                + (model.transfunc.dfdS(m, S) * m.matmul(self.A_grid[idx].transpose(-1, -2)).unsqueeze(-1).unsqueeze(-1)).sum(2, keepdim=True) \
                + (model.transfunc.ddfdxdS(m, S) * self.A_grid[idx].matmul(S).unsqueeze(-1).unsqueeze(-1)).sum(1, keepdim=True).sum(2, keepdim=True)

            dEdS_grid_asym = self.A_grid[idx].transpose(-1, -2).matmul(model.transfunc.dfdx(m, S)).unsqueeze(1).unsqueeze(1) \
                + 0.5 * self.A_grid[idx].transpose(-1, -2).matmul(self.A_grid[idx]).unsqueeze(1).unsqueeze(1)

            dEdS_grid = dEdS_grid_sym + dEdS_grid_asym + dEdS_grid_asym.transpose(-2, -1) - batch_make_diag(dEdS_grid_asym)  # account for symmetry in S

        # return more compact representation of gradients
        return dEdm_grid.squeeze(1).squeeze(1), dEdS_grid.squeeze(1).squeeze(1)

    def get_ExpectedLogLike_grad(self, model, m, S, idx):
        # gradients of expected log likelihood with respect to mean and covariance functions
        # predict mean and variance given current function values
        with torch.no_grad():
            mu = model.outputMapping.OutMean(m, S)
            cov = model.outputMapping.OutCov(m, S)

            # get gradients of output mapping
            dmudm, dcovdm = model.outputMapping.dOutdm(m, S)  # T x D x K
            dmudS, dcovdS = model.outputMapping.dOutdS(m, S)  # T x D x K x K

            # get gradient of likelihood: all are T x 1 x D
            dLdmu, dLdmu_jump, dLdcov, dLdcov_jump = model.like.expected_loglik_gradients(mu, cov, idx)

            # use chain rule to compute gradients of log likelihood wrt m and S
            dLdm = dLdmu.matmul(dmudm) + dLdcov.matmul(dcovdm)  # T x 1 x K

            # dmus = T x 1 x K and dSs = T x K x K
            dLdS = (dLdmu.unsqueeze(-1).unsqueeze(-1) * dmudS.unsqueeze(1)).sum(2).squeeze(1) \
                + (dLdcov.unsqueeze(-1).unsqueeze(-1) * dcovdS.unsqueeze(1)).sum(2).squeeze(1)

            dLdm_jump = dLdmu_jump.matmul(dmudm) + dLdcov_jump.matmul(dcovdm)  # T x 1 x K

            dLdS_jump = (dLdmu_jump.unsqueeze(-1).unsqueeze(-1) * dmudS.unsqueeze(1)).sum(2).squeeze(1) \
                + (dLdcov_jump.unsqueeze(-1).unsqueeze(-1) * dcovdS.unsqueeze(1)).sum(2).squeeze(1)

        return dLdm, dLdm_jump, dLdS, dLdS_jump

    def get_initialState(self, mu0, V0):
        # get initial state from current values
        m0 = []
        S0 = []
        for idx in range(self.nTrials):
            m0.append(mu0[idx].detach() - torch.matmul(self.eta[idx][0, ], V0[idx].detach().permute(1, 0)))
            S0.append(torch.inverse(2 * self.Psi[idx][0, ] + torch.inverse(V0[idx].detach())))

        self.initialMean = m0  # R list with 1 x K
        self.initialCov = S0  # R list with K x K

    def solveForward_GaussMarkov_grid(self, m0, S0, idx):
        # solves differential equations using Forward Euler method on pre-defined grid

        # initialise variables
        m_grid = torch.zeros_like(self.eta[idx]).type(float_type)  # T x 1 x K
        S_grid = torch.zeros_like(self.Psi[idx]).type(float_type)  # T x K x K

        # get total number of grid points
        T = m_grid.shape[0]

        # set initial state
        m_grid[0, :, :] = m0[:]
        S_grid[0, :, :] = S0[:]

        # Forward Euler to solver ODEs
        for tt in range(T - 1):
            m_grid[tt + 1, :, :] = m_grid[tt, :, :] - self.dtstep * (torch.matmul(m_grid[tt, :, :], self.A_grid[idx][tt, :, :].transpose(-2, -1)) - self.b_grid[idx][tt, :, :])

            S_grid[tt + 1, :, :] = S_grid[tt, :, :] - self.dtstep * (torch.matmul(self.A_grid[idx][tt, :, :], S_grid[tt, :, :]) +
                                                                     torch.matmul(S_grid[tt, :, :], self.A_grid[idx][tt, :, :].transpose(-2, -1)) -
                                                                     torch.eye(self.nLatent).type(float_type).unsqueeze(0))
        # symmetrize solution to improve numerical stability
        S_grid = 0.5 * (S_grid + S_grid.transpose(-2, -1))

        return m_grid, S_grid

    def predict_marginals(self, idx, t):
        # function to evaluate GP at new points for trial idx
        # solve for m, S on grid
        m, S = self.solveForward_GaussMarkov_grid(self.initialMean[idx], self.initialCov[idx], idx)

        # interpolate points off grid
        T = t.size()[0]
        m_linp = torch.zeros(T, 1, self.nLatent).type(float_type)
        S_linp = torch.zeros(T, self.nLatent, self.nLatent).type(float_type)
        for i in range(T):
            m_linp[i, :, :] = linInterp(t[i], m, self.t_grid[idx])
            S_linp[i, :, :] = linInterp(t[i], S, self.t_grid[idx])

        return m_linp, S_linp

    def predict_conditionalParams(self, idx, t):
        # interpolate points off grid
        T = t.size()[0]
        b_linp = torch.zeros(T, 1, self.nLatent).type(float_type)
        A_linp = torch.zeros(T, self.nLatent, self.nLatent).type(float_type)
        for i in range(T):
            b_linp[i, :, :] = linInterp(t[i], self.b_grid[idx], self.t_grid[idx])
            A_linp[i, :, :] = linInterp(t[i], self.A_grid[idx], self.t_grid[idx])

        return A_linp, b_linp

    def update_GaussMarkov(self, model, m, S, idx):
        # update A(t), b(t) lambda functions
        with torch.no_grad():
            A_grid_new = - model.transfunc.dfdx(m, S) + 2 * self.Psi[idx]
            b_grid_new = model.transfunc.f(m, S) + m.matmul(A_grid_new.transpose(-2, -1)) - self.eta[idx]

        # compute criterion for convergence
        A_diff_sq_norm = ((self.A_grid[idx] - A_grid_new)**2).sum()
        b_diff_sq_norm = ((self.b_grid[idx] - b_grid_new)**2).sum()

        if (A_diff_sq_norm < self.convergenceTol) & (b_diff_sq_norm < self.convergenceTol):
            convergence_flag = True
        else:
            convergence_flag = False

        # update parameters towards fixed point update
        self.A_grid[idx] = self.A_grid[idx] - self.learningRate * (self.A_grid[idx] - A_grid_new)
        self.b_grid[idx] = self.b_grid[idx] - self.learningRate * (self.b_grid[idx] - b_grid_new)

        return convergence_flag

    def run_inference_single(self, model, idx):
        # run variational inference for a single trial

        m, S = self.solveForward_GaussMarkov_grid(self.initialMean[idx], self.initialCov[idx], idx)

        dLdm, dLdm_jump, dLdS, dLdS_jump = self.get_ExpectedLogLike_grad(model, m, S, idx)

        self.solveBackward_LagrangeMultipliers(model, m, S, dLdm, dLdm_jump, dLdS, dLdS_jump, idx)

        convergence_flag = self.update_GaussMarkov(model, m, S, idx)

        return convergence_flag

    def run_inference(self, model, niter=10):

        # update initial state value
        self.get_initialState(model.initialMean, model.initialCov)

        # otherwise run simple for loop over trials
        for idx in range(self.nTrials):
            for _ in range(niter):

                # run inference loops for ith trial
                convergence_flag = self.run_inference_single(model, idx)

                # break loop if trial has converged
                if convergence_flag is True:
                    break
