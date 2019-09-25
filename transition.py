import torch
from torch import nn
from utils import apply_along_axis, unsqueeze_as, get_all_grads, create_copy_with_grad, logdet, stack_along_dim, batch_inverse, batch_diag, batch_make_diag, batch_inverse_psd, batch_vec_to_diag
from settings import float_type, jitter_level, var_init, mean_init
import warnings

# --------------- base class for transition functions ---------------


class TransitionFunction(nn.Module):
    def __init__(self):
        super(TransitionFunction, self).__init__()
        """
        transition function class

        for q(x) ~ N(m,S) we need the expectations

            - f = E[f(x(t))]
            - ff = E[f(x)'f(x)]
            - dfdx = E[df/dx]

        we also need function to compute the gradients needed for inference:

        for q(x) ~ N(m,S) we need the expectations
            - d/dS, d/dm E[f(x(t))]
            - d/dS, d/dm E[f(x)'f(x)]
            - d/dS, d/dm E[df/dx]

        Note that the gradients wrt S need to take into account that S is SYMMETRIC, e.g. d/dS trace(A S) = A + A' - diag(A)
        the rest of the code expects this to be taken care of

        Input Dimensions:
            In order to make this code work seamlessly with
            kernel implementations (that expect R (batch size) x N (num points) x K (latent dim) objects as inputs),
            we expect:
            >  m - R x 1 x K
            >  S - R x K x K (S never directly enters kernels, but gets used as a batch x K x K matrix in torch.bmm)

        Gradient computations:
            By default all the gradients are computed via utils.get_all_grads(),
            which takes an object Output, and computes first batchwise, then entrywise the
            gradient @ Output[batch][i] / @ Input[batch], resulting in a gradient the same shape as Input[batch],
            for each element of Output.

            Therefore the gradient's shape is
            [R, Output.shape[1:], Input.shape[1:]]
        """

    def closedFormUpdates(self, mq, Sq, Aq, bq, trLen, wwLeg):
        # some transition functions allow for closed form updates, implement them here
        # variables that use closed form updates shouldn't be included as torch parameters
        pass

    def refresh_stored_values(self):
        pass

    def log_prior_distribution(self):
        # for cost function computation
        return torch.zeros(1, 1).type(float_type)

    def f(self, m, S):
        """ Outputs R x 1 x K
        """
        raise NotImplementedError()

    def ff(self, m, S):
        """ Outputs R x 1 x 1
        """
        raise NotImplementedError()

    def dfdx(self, m, S):
        """ Outputs (R x K x K)
        """
        raise NotImplementedError()

    # gradients wrt covariance of x(t)
    def dfdS(self, m, S):
        """ Outputs (R x 1 x K) x (K x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.f(m, S)
        return get_all_grads(target, S)[0]

    def dffdS(self, m, S):
        """ Outputs (R x 1 x 1) x (K x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.ff(m, S)
        return get_all_grads(target, S)[0]

    def ddfdxdS(self, m, S):
        """ Outputs (R x 1 x 1) x (K x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.dfdx(m, S)
        return get_all_grads(target, S)[0]

    # gradients wrt mean of x(t)
    def dfdm(self, m, S):
        """ Outputs (R x 1 x K) x (1 x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.f(m, S)
        return get_all_grads(target, m)[0]

    def dffdm(self, m, S):
        """ Outputs (R x 1 x 1) x (1 x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.ff(m, S)
        return get_all_grads(target, m)[0]

    def ddfdxdm(self, m, S):
        """ Outputs (R x K x K) x (1 x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.dfdx(m, S)
        return get_all_grads(target, S)[0]

    # forward pass of model
    def forward(self, m, S):
        f = self.f(m, S)
        ff = self.ff(m, S)
        dfdx = self.dfdx(m, S)
        prior = self.log_prior_distribution()
        return f, ff, dfdx, prior


# --------------- stationary linear transition function ---------------
class LinearTransition(TransitionFunction):
    def __init__(self, A, b, useClosedForm=True):
        super(LinearTransition, self).__init__()
        # f(x) = - x A' + b --> dynamics operate on a column vector, transpose operates on a row vector
        self.useClosedForm = useClosedForm
        if self.useClosedForm:
            self.Dynamics = A.type(float_type)  # K x K
            self.Offset = b.type(float_type)  # 1 x K
        else:
            self.Dynamics = nn.Parameter(A.type(float_type))  # K x K
            self.Offset = nn.Parameter(b.type(float_type))  # 1 x K
        # input is
        # m = R x 1 x K
        # S = R x K x K

    def closedFormUpdates(self, mq, Sq, Aq, bq, trLen, wwLeg):
        # closed form update for linear stationary transition function
        # input dims:
        # mq is R x T x 1 x K
        # Sq is R x T x K x K
        # trLen is length R list
        # wwLeg is R x T

        if self.useClosedForm:
            with torch.no_grad():
                numTrials = len(trLen)  # number of trials

                # sum_R int_T dt < X X^T> = mq mq^T + S
                XX = (mq.transpose(-2, -1).matmul(mq) + Sq)
                AutoCorr = (XX * wwLeg.unsqueeze(-1).unsqueeze(-1)).sum(0).sum(0)  # K x K autocorrelation
                XCorr = (Aq.matmul(XX) * wwLeg.unsqueeze(-1).unsqueeze(-1)).sum(0).sum(0)  # K x K crosscorrelation
                OffsetCorr = ((self.Offset - bq).transpose(-2, -1).matmul(mq) * wwLeg.unsqueeze(-1).unsqueeze(-1)).sum(0).sum(0)  # K x K input correlation

                Anew, _ = torch.solve((XCorr + OffsetCorr).transpose(-2, -1), AutoCorr.transpose(-2, -1))
                bnew = ((bq + mq.matmul(Anew - Aq.transpose(-2, -1))) / torch.tensor(trLen).type(float_type).view(-1, 1, 1, 1) * wwLeg.unsqueeze(-1).unsqueeze(-1)).sum(0).sum(0)

                self.Dynamics = Anew.transpose(-2, -1)
                self.Offset = bnew / numTrials
        else:
            pass

    def f(self, m, S):
        # f(x) = -m' * A' + b'
        # R x 1 x K
        return - m.matmul(self.Dynamics.permute(1, 0)) + self.Offset

    def ff(self, m, S):
        # R x (T x 1 x 1)
        ff = (torch.matmul(self.Dynamics.permute(1, 0), self.Dynamics) * S).sum(-2, keepdim=True).sum(-1, keepdim=True) \
            + (m.matmul(self.Dynamics.permute(1, 0))**2).sum(-1, keepdim=True) \
            + (self.Offset * self.Offset).sum(-1, keepdim=True) \
            - 2 * (m.matmul(self.Dynamics.permute(1, 0)) * self.Offset).sum(-1, keepdim=True)
        return ff  # R x 1 x 1

    def dfdx(self, m, S):
        # R x (T x K x K)
        # permute(1, 0) or not? same as dfdm, check
        return -self.Dynamics.unsqueeze(0)  # Jacobian: columns contain gradient wrt ith element

    # gradients wrt covariance of x(t)
    def dfdS(self, m, S):
        # (R x T x K) x (K x K)
        with torch.no_grad():
            out = torch.zeros(S.size()[0], m.size()[1], m.size()[2], S.size()[1], S.size()[2])
        return out

    def dffdS(self, m, S):
        # (R x T x 1) x (K x K)
        with torch.no_grad():
            out = self.Dynamics.permute(1, 0).matmul(self.Dynamics).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return out + out.transpose(-2, -1) - batch_make_diag(out)

    def ddfdxdS(self, m, S):
        # R x (T x K x K) x (K x K)
        with torch.no_grad():
            out = torch.zeros(S.size()[0], S.size()[1], S.size()[2], S.size()[1], S.size()[2])
        return out

    # gradients wrt mean of x(t)
    def dfdm(self, m, S):
        # (R x T x K) x (1 x K)
        with torch.no_grad():
            out = -self.Dynamics.unsqueeze(0).unsqueeze(-1).permute(0, 1, 3, 2).unsqueeze(0)
        return out

    def dffdm(self, m, S):
        # (R x T x 1) x (1 x K)
        with torch.no_grad():
            df = 2 * m.matmul(self.Dynamics.permute(1, 0).matmul(self.Dynamics)) - 2 * self.Offset.matmul(self.Dynamics)
        return df.unsqueeze(1).unsqueeze(1)  # (R x 1 x 1) x (1 x K)

    def ddfdxdm(self, m, S):
        # R x (T x K x K) x (1 x K)
        with torch.no_grad():
            out = torch.zeros(m.size()[0], m.size()[2], m.size()[2], m.size()[1], m.size()[2])
        return out


# --------------- double well Transition function ---------------
class DoubleWell(TransitionFunction):
    def __init__(self):
        super(DoubleWell, self).__init__()

    def f(self, m, S):
        return 4 * (m - m**3 - 3 * m * S)  # R x 1 x K

    def ff(self, m, S):
        x2 = m**2 + S
        x4 = m**4 + 6 * m**4 * S + 3 * S**2
        x6 = m**6 + 15 * m**4 * S + 45 * m**2 * S**2 + 15 * S**3
        return 16 * (x2 - 2 * x4 + x6)  # R x 1 x 1

    def dfdx(self, m, S):
        return 4 - 12 * (m**2 + S)  # R x K x K

    # gradients wrt covariance of x(t)
    def dfdS(self, m, S):
        return 12 * m.unsqueeze(-1).unsqueeze(-1)

    def dffdS(self, m, S):
        dx2 = torch.ones_like(S)
        dx4 = 6 * m**4 + 6 * S
        dx6 = 15 * m**4 + 90 * m**2 * S + 45 * S**2
        return 16 * (dx2 - 2 * dx4 + dx6).unsqueeze(-1).unsqueeze(-1)

    def ddfdxdS(self, m, S):
        return torch.tensor(12.).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def dfdm(self, m, S):
        return (4 - 12 * m**2 - 3 * S).unsqueeze(-1).unsqueeze(-1)

    def dffdm(self, m, S):
        dx2 = 2 * m
        dx4 = 4 * m**3 + 24 * m**3 * S
        dx6 = 6 * m**5 + 60 * m**3 * S + 90 * m * S**2
        return 16 * (dx2 - 2 * dx4 + dx6).unsqueeze(-1).unsqueeze(-1)

    def ddfdxdm(self, m, S):
        return 24 * m.unsqueeze(-1).unsqueeze(-1)


# --------------- Gaussian Process Transition functions ---------------
class SparseGP(TransitionFunction):
    def __init__(self, kern, Zs, useClosedForm=True):
        super(SparseGP, self).__init__()
        """
        sparse GP transition function
        kern -- kernel module
        Zs   -- torch tensor of nZ x D with inducing point locations: will be converted to torch parameter
        """
        self.useClosedForm = useClosedForm
        self.kern = kern
        self.Zs = nn.Parameter(Zs.type(float_type))  # M x K
        numZ, xDim = Zs.size()
        self.numZ = numZ  # M number of inducing ponts
        self.xDim = xDim  # K latent dimensionality
        self.jitter_level = jitter_level
        self.initialise_inducingPoints()
        self.store_Kzz_inv()

    def closedFormUpdates(self, mq, Sq, Aq, bq, trLen, wwLeg):
        # closed form update for inducing point distribution
        if self.useClosedForm:
            with torch.no_grad():
                # get matrices we need
                Kzz = self.get_Kzz()  # 1 x M x M
                Kzxxz = self.get_E_Kzxxz(mq, Sq)  # V x R x M x M
                Kxz = self.get_E_Kxz(mq, Sq)  # V x R x 1 x M
                dKxz = self.get_E_dKxz(mq, Sq)  # V x R x 1 x M x K

                # update for q_sigma (same across all dimensions)
                CovMat = Kzz + (Kzxxz * wwLeg.unsqueeze(-1).unsqueeze(-1)).sum(0).sum(0)  # M x M
                CovMatinvK, _ = torch.solve(Kzz, CovMat)  # (Kzz + Kzxxz)^{-1} Kzz
                self.q_sigma = (Kzz.matmul(CovMatinvK)).repeat(self.xDim, 1, 1)  # Kzz (Kzz + Kzxxz)^{-1} Kzz  repeated for each dimension K x M x M

                # update for inducing points q_mu
                Xcorr1 = (Kxz.transpose(-2, -1).matmul(- mq.matmul(Aq.transpose(-2, -1)) + bq) * wwLeg.unsqueeze(-1).unsqueeze(-1)).sum(0).sum(0)
                Xcorr2 = (dKxz.transpose(-2, -3).squeeze(-2).matmul(Sq).matmul(Aq.transpose(-2, -1)) * wwLeg.unsqueeze(-1).unsqueeze(-1)).sum(0).sum(0)
                q_mu_new, _ = torch.solve(Xcorr1 - Xcorr2, CovMat)
                self.q_mu = Kzz.matmul(q_mu_new).squeeze(0)
        else:
            pass

    def refresh_stored_values(self):
        self.store_Kzz_inv()

    def log_prior_distribution(self):
        # for cost function computation
        return - self.kullback_leibler()

    def store_Kzz_inv(self):
        # store inverse for inference operations
        with torch.no_grad():
            Kzz_inv = self.get_Kzz_inv()
        self.Kzz_inv = Kzz_inv

    # ----------------------- Functions that need updating to use different GP model -----------------------

    def initialise_inducingPoints(self):
        if self.useClosedForm:
            # don't include as parameters if closed form updating
            self.q_mu = mean_init * torch.ones(self.numZ, self.xDim).type(float_type)  # M x K posterior mean
            self.q_sigma = var_init * torch.ones(self.numZ, self.xDim).type(float_type)  # M x K diagonal posterior standard deviation
        else:
            self.q_mu = nn.Parameter(mean_init * torch.ones(self.numZ, self.xDim).type(float_type))  # M x K posterior mean
            self.q_sigma = nn.Parameter(var_init * torch.ones(self.numZ, self.xDim).type(float_type))  # M x K diagonal posterior standard deviation

    def get_Kzz(self):
        Kzz = self.kern(mode="k", x1=self.Zs.unsqueeze(0), x2=self.Zs.unsqueeze(0))
        Kzz = Kzz + (self.jitter_level * torch.eye(self.numZ, device=Kzz.device).unsqueeze(0)).type(float_type)
        return Kzz

    def get_Kzz_inv(self):
        # output is 1 x M x M
        Kzz = self.get_Kzz()
        Kzz_inv = batch_inverse(Kzz)
        return Kzz_inv

    def get_Kxz(self, x):
        return self.kern(mode="k", x1=x, x2=self.Zs.unsqueeze(0))

    def get_E_Kzxxz(self, m, S):
        Kzxxz = self.kern(mode="psi2", x2=unsqueeze_as(self.Zs, m), mu=m, cov=S)
        # return Kzxxz + (self.jitter_level**2 * torch.eye(self.numZ, device=Kzxxz.device).unsqueeze(0))  # V x R x M x M
        return Kzxxz

    def get_E_Kxz(self, m, S):
        return self.kern(mode="psi1", x2=unsqueeze_as(self.Zs, m), mu=m, cov=S)

    def get_E_dKxz(self, m, S):
        return self.kern(mode="psid1", x2=unsqueeze_as(self.Zs, m), mu=m, cov=S)  # R x 1 x M x K

    def get_dm_E_Kzxxz(self, m, S):
        return self.kern.dPsi2dmu(x2=unsqueeze_as(self.Zs, m), mu=m, cov=S)  # (R x M x M) x 1 x K

    def get_dm_E_Kxz(self, m, S):
        return self.kern.dPsi1dmu(x2=unsqueeze_as(self.Zs, m), mu=m, cov=S)  # (R x N1 x N2) x (1 x K)

    def get_dm_E_dKxz(self, m, S):
        return self.kern.dPsid1dmu(x2=unsqueeze_as(self.Zs, m), mu=m, cov=S)  # (R x 1 x M x K) x (1 x K)

    def get_dS_E_Kzxxz(self, m, S):
        return self.kern.dPsi2dcov(x2=unsqueeze_as(self.Zs, m), mu=m, cov=S)   # (R x M x M) x K x K

    def get_dS_E_Kxz(self, m, S):
        return self.kern.dPsi1dcov(x2=unsqueeze_as(self.Zs, m), mu=m, cov=S)  # (R x N1 x N2) x (K x K)

    def get_dS_E_dKxz(self, m, S):
        return self.kern.dPsid1dcov(x2=unsqueeze_as(self.Zs, m), mu=m, cov=S)  # (R x 1 x M x K) x (K x K)

    def assemble_qs(self):
        # returns M x K q_mu
        # K x M x M q_sigma
        q_mu = self.q_mu
        if self.q_sigma.view(self.numZ, self.xDim, -1).size(-1) == 1:  # diagonal case
            q_sigma = batch_vec_to_diag(self.q_sigma.transpose(-2, -1))  # K x M x M
        else:
            q_sigma = self.q_sigma
        return q_mu, q_sigma

    # ----------------------- core functions needed for inference that should work give the above -----------------------

    def predict(self, x):
        """
        predict conditional mean of sparse GP at N new input locations
        x is R x N x K
        """
        Kxz = self.get_Kxz(x)
        Kzz = self.get_Kzz()
        q_mu, q_sigma = self.assemble_qs()
        Ak, _ = torch.solve(q_mu, Kzz)
        mean = Kxz.matmul(Ak)
        # pdb.set_trace()
        # Kzz_inv = self.get_Kzz_inv()
        # mean = Kxz.matmul(Kzz_inv).matmul(q_mu)

        # pdb.set_trace()
        # Kxx = self.kern(mode="kdiag", x1=x)
        # cov = Kxx + torch.sum(Ak * (((q_sigma - Kzz).bmm(Ak.permute(0, 2, 1))).permute(0, 2, 1)), dim=2).unsqueeze(2)
        # return mean, cov
        return mean

    def f(self, m, S):
        """
        predict expected conditional mean of sparse GP
        """
        Kxz = self.get_E_Kxz(m, S)
        Kzz = self.get_Kzz()
        q_mu, q_sigma = self.assemble_qs()
        Ak, _ = torch.solve(q_mu, Kzz)
        mean = Kxz.matmul(Ak)
        return mean

    def ff(self, m, S):
        #  this should be sum_i <f_i^2>
        # output is R x 1 x 1

        Kxx = self.kern(mode="psi0", mu=m, cov=S)  # < k(x,x)> # R x 1
        # Kzz_inv = self.get_Kzz_inv()
        Kzz = self.get_Kzz()
        Kzxxz = self.get_E_Kzxxz(m, S)  # V x R x M x M
        q_mu, q_sigma = self.assemble_qs()
        Kzz_inv_U, _ = torch.solve(q_mu.unsqueeze(0), Kzz)

        # Kzz^{-1} sum(Su)
        Kzz_inv_q_sigmas, _ = torch.solve(q_sigma.sum(0), Kzz)

        # Kzz^{-1} Kzxxz
        Kzz_inv_Kzxxz, _ = torch.solve(Kzxxz, Kzz)

        # Kzz^{-1} Kzxxz Kzz^{-1}
        term1 = self.xDim * (Kxx - batch_diag(Kzz_inv_Kzxxz).sum(-1, keepdim=True)).unsqueeze(-1)  # (R x 1 x 1)
        # trace ( Kzz^{-1} sum(Su) Kzz^{-1} Kzxxz ) = vec ( sum(Su) Kzz^{-1}) * vec( Kzz^{-1} Kzxxz )
        term2 = (Kzz_inv_q_sigmas.transpose(-2, -1) * Kzz_inv_Kzxxz).sum(-2, keepdim=True).sum(-1, keepdim=True)
        term3 = (Kzxxz.matmul(Kzz_inv_U) * Kzz_inv_U).sum(-2, keepdim=True).sum(-1, keepdim=True)
        ff = term1 + term2 + term3

        if torch.any(ff < 0):
            warnings.warn('negative variance in expected transition function encountered')

        return ff

    def dfdx(self, m, S):
        """
            We are using the identity < df/dx > = < f(x) (x-m)^T > S^-1
            Formula is from archembeaua07, between eq 20-21

            It can be proved by expanding
                < df/dx > = integral df/dx p(x) dx
                Then doing integration by parts (using @p(x)/@x, with p(x) = N_x(m, S))

        output is R x K x K where gradient is computed as grad of column vector wrt column vector
        """
        Kzz = self.get_Kzz()
        dKxz = self.get_E_dKxz(m, S)  # R x 1 x M x Kgrad
        q_mu, q_sigma = self.assemble_qs()
        Kzz_inv_U, _ = torch.solve(q_mu.unsqueeze(0), Kzz)
        df = dKxz.transpose(-1, -3).transpose(-1, -2).matmul(Kzz_inv_U).squeeze(-2)  # ( R x Kgrad x 1 x M ) * (R x 1 x M x K)  = R x Kgrad x 1 x K, squeeze out 1 dim

        return df.transpose(-2, -1)  # transpose such that grad is along columns

    # gradients wrt covariance of x(t)
    def dfdS(self, m, S):
        """ Outputs (R x 1 x K) x (K x K)
        """
        with torch.no_grad():
            Kxz = self.get_dS_E_Kxz(m, S)  # (R x N1 x N2) x (K x K)
            Kzz = self.get_Kzz()
            # Kzz_inv = self.Kzz_inv  # 1 x M x M use stored inverse since Zs not updated across inference iterations
            q_mu, q_sigma = self.assemble_qs()
            # Ak = (Kxz.transpose(2, 1) * Kzz_inv.unsqueeze(-1).unsqueeze(-1)).sum(1, keepdim=True)  # (R x 1 x M) x K x K
            # mean = (Ak.transpose(2, 1) * (q_mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))).sum(1, keepdim=True)  # (R x 1 x K) x K x K
            Ak, _ = torch.solve(Kxz.permute(0, 3, 4, 2, 1), Kzz)
            mean = Ak.transpose(-2, -1).matmul(q_mu).permute(0, 3, 4, 1, 2)
        return mean

    def dffdS(self, m, S):
        """ Outputs (R x 1 x 1) x (K x K)
        """
        with torch.no_grad():
            Kxx = self.kern.dPsi0dcov(mu=m, cov=S).detach()  # < k(x,x)> # (R x 1) x K x K
            Kzz = self.get_Kzz()
            Kzxxz = self.get_dS_E_Kzxxz(m, S)  # (R x M x M) x K x K
            q_mu, q_sigma = self.assemble_qs()  # q_sigma = q_diag.pow(2).diag()
            Kzz_inv_U, _ = torch.solve(q_mu.unsqueeze(0), Kzz)  # R x M x K
            Kzz_inv_q_sigmas, _ = torch.solve(q_sigma.sum(0), Kzz)  # Kzz^{-1} sum(Su)

            # Kzz^{-1} Kzxxz
            Kzz_inv_Kzxxz, _ = torch.solve(Kzxxz.permute(0, 3, 4, 1, 2), Kzz)  # R x (K x K) x (M x M)

            term1 = self.xDim * (Kxx.unsqueeze(-3) - batch_diag(Kzz_inv_Kzxxz).sum(-1, keepdim=True).unsqueeze(-1).permute(0, 3, 4, 1, 2))  # (R x 1 x 1) x (K x K)
            # trace ( Kzz^{-1} sum(Su) Kzz^{-1} Kzxxz ) = vec ( sum(Su) Kzz^{-1}) * vec( Kzz^{-1} Kzxxz )
            term2 = (Kzz_inv_q_sigmas.transpose(-2, -1).unsqueeze(-3).unsqueeze(-3) * Kzz_inv_Kzxxz).sum(-2, keepdim=True).sum(-1, keepdim=True).permute(0, 3, 4, 1, 2)
            term3 = (Kzxxz.permute(0, 3, 4, 1, 2).matmul(unsqueeze_as(Kzz_inv_U, Kzxxz, dim=0)) * unsqueeze_as(Kzz_inv_U, Kzxxz, dim=0)).sum(-2, keepdim=True).sum(-1, keepdim=True).permute(0, 3, 4, 1, 2)

            dff = term1 + term2 + term3

            # term4 = Kzz_inv_Kzxxz.sum(-2, keepdim=True).sum(-1, keepdim=True).permute(0, 3, 4, 1, 2)
            # dff = term2

        return dff

    def ddfdxdS(self, m, S):
        """ Outputs (R x 1 x 1) x (K x K)
        """
        with torch.no_grad():
            # Kzz_inv = self.Kzz_inv  # 1 x M x M
            Kzz = self.get_Kzz()
            dKxz = self.get_dS_E_dKxz(m, S)  # (R x 1 x M x K) x (K x K)
            q_mu, q_sigma = self.assemble_qs()
            # Kzz_inv_U = Kzz_inv.matmul(q_mu.unsqueeze(0))  # R x M x K
            Kzz_inv_U, _ = torch.solve(q_mu.unsqueeze(0), Kzz)  # R x M x K
            df = (dKxz * Kzz_inv_U.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).transpose(1, 3)).sum(2)  # ( R x 1 x M x Kgrad ) (K x K) * (R x K x M x 1) (1 x 1)  = (R x K x Kgrad) x ( K x K), squeeze out 1 dim
        return df  # transpose such that grad is along columns

    # gradients wrt mean of x(t)
    def dfdm(self, m, S):
        """ Outputs (R x 1 x K) x (1 x K)
        """
        with torch.no_grad():
            Kxz = self.get_dm_E_Kxz(m, S)  # (R x N1 x N2) x (1 x K)
            Kzz = self.get_Kzz()
            # Kzz_inv = self.Kzz_inv  # 1 x M x M
            q_mu, q_sigma = self.assemble_qs()
            Ak, _ = torch.solve(Kxz.permute(0, 3, 4, 2, 1), Kzz)
            mean = Ak.transpose(-2, -1).matmul(q_mu).permute(0, 3, 4, 1, 2)

        return mean

    def dffdm(self, m, S):
        """ Outputs (R x 1 x 1) x (1 x K)
        """
        with torch.no_grad():
            Kxx = self.kern.dPsi0dmu(mu=m, cov=S).detach()  # < k(x,x)> # (R x 1) x K x K
            Kzz = self.get_Kzz()
            Kzxxz = self.get_dm_E_Kzxxz(m, S)  # (R x M x M) x K x K
            q_mu, q_sigma = self.assemble_qs()
            Kzz_inv_U, _ = torch.solve(q_mu.unsqueeze(0), Kzz)  # R x M x K
            Kzz_inv_q_sigmas, _ = torch.solve(q_sigma.sum(0), Kzz)  # Kzz^{-1} sum(Su)

            # Kzz^{-1} Kzxxz
            Kzz_inv_Kzxxz, _ = torch.solve(Kzxxz.permute(0, 3, 4, 1, 2), Kzz)  # R x (K x K) x (M x M)
            term1 = self.xDim * (Kxx.unsqueeze(-3) - batch_diag(Kzz_inv_Kzxxz).sum(-1, keepdim=True).unsqueeze(-1).permute(0, 3, 4, 1, 2))  # (R x 1 x 1) x (K x K)
            # trace ( Kzz^{-1} sum(Su) Kzz^{-1} Kzxxz ) = vec ( sum(Su) Kzz^{-1}) * vec( Kzz^{-1} Kzxxz )
            term2 = (Kzz_inv_q_sigmas.transpose(-2, -1).unsqueeze(-3).unsqueeze(-3) * Kzz_inv_Kzxxz).sum(-2, keepdim=True).sum(-1, keepdim=True).permute(0, 3, 4, 1, 2)
            term3 = (Kzxxz.permute(0, 3, 4, 1, 2).matmul(unsqueeze_as(Kzz_inv_U, Kzxxz, dim=0)) * unsqueeze_as(Kzz_inv_U, Kzxxz, dim=0)).sum(-2, keepdim=True).sum(-1, keepdim=True).permute(0, 3, 4, 1, 2)
            dff = term1 + term2 + term3

            # term4 = Kzz_inv_Kzxxz.sum(-2, keepdim=True).sum(-1, keepdim=True).permute(0, 3, 4, 1, 2)
            # dff = term2
        return dff

    def ddfdxdm(self, m, S):
        """ Outputs (R x K x K) x (1 x K)
        """
        with torch.no_grad():
            # Kzz_inv = self.Kzz_inv  # 1 x M x M
            Kzz = self.get_Kzz()

            dKxz = self.get_dm_E_dKxz(m, S)  # (R x 1 x M x K) x (1 x K)
            q_mu, q_sigma = self.assemble_qs()
            # Kzz_inv_U = Kzz_inv.matmul(q_mu.unsqueeze(0))  # R x M x K
            Kzz_inv_U, _ = torch.solve(q_mu.unsqueeze(0), Kzz)  # R x M x K

            # pdb.set_trace()
            df = (dKxz * Kzz_inv_U.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).transpose(1, 3)).sum(2)  # ( R x 1 x M x Kgrad ) ( 1 x K) * (R x K x M x 1) ( 1 x 1)  = (R x K x Kgrad) x ( 1 x K), squeeze out 1 dim
            # df = dKxz.permute(0, 1, 4, 5, 3, 2).matmul(Kzz_inv_U).permute(0, 1, 5, 4, 2, 3)
            # ( R x 1 x M x Kgrad ) ( 1 x K) ---> ( R x 1) x ( 1 x K) x (Kgrad x M) * ( M x K)  = (R x 1 x K x Kgrad) x ( 1 x K)

        return df  # transpose such that grad is along columns
        # return dKxz

    def kullback_leibler(self):
        # Kullback leibler divergence between inducing points posterior and prior
        q_mu, q_sigma = self.assemble_qs()  # M x K and M x K x K
        # q_sigma = q_diag.pow(2).sum(-1).diag()
        numZ = q_mu.size(0)

        Kzz = self.get_Kzz()  # 1 x M x M

        # trace( Kzz_inv * q_sigma)
        Kzzinv_sig, _ = torch.solve(q_sigma.sum(0), Kzz)
        tt1 = batch_diag(Kzzinv_sig).sum()

        # q_mu' * Kzz_inv * q_mu
        Kzzinv_q, _ = torch.solve(q_mu, Kzz)

        tt2 = (Kzzinv_q * q_mu).sum()

        # logdet(Kzz) - logdet(q_sigma) - k
        tt3 = self.xDim * apply_along_axis(logdet, Kzz, dim=0).sum()

        tt4 = - apply_along_axis(logdet, q_sigma, dim=0).sum() - self.xDim * numZ

        # sum everything up
        kld = torch.sum(0.5 * (tt1 + tt2 + tt3 + tt4))

        assert(torch.all(kld >= 0))  # make sure KLD is non-zero

        return kld

    # autograd version of gradients for debugging:
    def dfdS_autograd(self, m, S):
        """ Outputs (R x 1 x K) x (K x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.f(m, S)
        grad = get_all_grads(target, S)[0]
        return grad + grad.transpose(-2, -1) - batch_make_diag(grad)

    def dffdS_autograd(self, m, S):
        """ Outputs (R x 1 x 1) x (K x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.ff(m, S)
        grad = get_all_grads(target, S)[0]
        return grad + grad.transpose(-2, -1) - batch_make_diag(grad)

    def ddfdxdS_autograd(self, m, S):
        """ Outputs (R x 1 x 1) x (K x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.dfdx(m, S)
        grad = get_all_grads(target, S)[0]
        return grad + grad.transpose(-2, -1) - batch_make_diag(grad)

    # gradients wrt mean of x(t)
    def dfdm_autograd(self, m, S):
        """ Outputs (R x 1 x K) x (1 x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.f(m, S)
        return get_all_grads(target, m)[0]

    def dffdm_autograd(self, m, S):
        """ Outputs (R x 1 x 1) x (1 x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.ff(m, S)
        return get_all_grads(target, m)[0]

    def ddfdxdm_autograd(self, m, S):
        """ Outputs (R x K x K) x (1 x K)
        """
        m, S = create_copy_with_grad(m, S)
        target = self.dfdx(m, S)
        return get_all_grads(target, m)[0]


class FixedPointSparseGP(SparseGP):
    def __init__(self, kern, Zs, Zs_fx, fixedpointARD=True, useClosedForm=True):
        # super(FixedPointSparseGP, self).__init__()
        nn.Module.__init__(self)
        """
        sparse second layer GP conditioned on fixed point observations and Jacobian observations
        kern  -- kernel module
        Zs    -- torch tensor of numZ x D with initial inducing point locations: will be converted to torch parameter
        Zs_fx -- torch tensor of numF x D with initial fixed point locations: will be converted to torch parameter
        """
        # hacky way of doing class construction to allow for calling method to store Kzz_inv in constructor
        self.useClosedForm = useClosedForm
        self.kern = kern
        self.Zs = nn.Parameter(Zs.type(float_type))  # M x K
        self.Zs_fx = nn.Parameter(Zs_fx.type(float_type))  # F x K
        numZ, xDim = Zs.size()
        numFx = Zs_fx.size(0)
        self.numZ = numZ  # M number of inducing ponts
        self.xDim = xDim  # K latent dimensionality
        self.numFx = numFx  # F number of fixed ponts
        self.jitter_level = jitter_level
        self.initialise_inducingPoints()
        self.initialise_Jacobians()
        self.initialise_fixedPointsUncertainty(fixedpointARD)
        self.store_Kzz_inv()

    def initialise_fixedPointsUncertainty(self, fixedpointARD):
        # initialise uncertainty over fixed point locations
        if fixedpointARD:
            var = var_init**(0.5) * torch.ones(self.Zs_fx.size(0), 1).type(float_type)
            self.Zs_fx_sqrt = nn.Parameter(var)  # optimise over uncertainty in fixed point locations
        else:
            var = torch.zeros(self.Zs_fx.size(0), 1).type(float_type)
            self.Zs_fx_sqrt = var  # treat fixed point locations as certain

    def initialise_Jacobians(self):
        if self.useClosedForm:
            self.q_Jfx = torch.zeros(self.numFx * self.xDim, self.xDim).type(float_type)  # local Jacobian at fixed point location
        else:
            self.q_Jfx = nn.Parameter(torch.zeros(self.numFx * self.xDim, self.xDim).type(float_type))  # local Jacobian at fixed point location

    def closedFormUpdates(self, mq, Sq, Aq, bq, trLen, wwLeg):
        # closed form update for inducing point distribution
        if self.useClosedForm:
            with torch.no_grad():
                # get matrices we need
                Kblk = self.get_Kzz()  # 1 x (M + F + FK) x (M + F + FK)
                Kzxxz = self.get_E_Kzxxz(mq, Sq)  # V x R x M x M
                Kxz = self.get_E_Kxz(mq, Sq)  # V x R x 1 x M
                dKxz = self.get_E_dKxz(mq, Sq)  # V x R x 1 x M x K

                # get blocks needed for gradient terms from KL[q(u)||p(u)]
                Kzz, Kcond, Kvec = self.get_conditional_blocks()
                Kzz_inv = torch.inverse(Kzz.squeeze(0)).unsqueeze(0)  # 1 x M x M
                Kcond_inv_Kvec, _ = torch.solve(Kvec.transpose(-1, -2), Kcond)
                Kcond_inv_Kvec_Kzz_inv = Kcond_inv_Kvec.matmul(Kzz_inv)

                KLmat_col1 = torch.cat([Kzz_inv, - Kcond_inv_Kvec_Kzz_inv], dim=1)
                KLmat_col2 = torch.cat([-Kcond_inv_Kvec_Kzz_inv.transpose(-2, -1),
                                        Kcond_inv_Kvec_Kzz_inv.matmul(Kcond_inv_Kvec.transpose(-2, -1))], dim=1)

                KLmat = torch.cat([KLmat_col1, KLmat_col2], dim=2)  # 1 x (M + F x FK) x (M + F x FK)

                # get integrated block matrices
                Kblk_inv_Kzxxz, _ = torch.solve((Kzxxz * wwLeg.unsqueeze(-1).unsqueeze(-1)).sum(0).sum(0), Kblk)
                Kblk_inv_Kzxxz_Kblk_inv, _ = torch.solve(Kblk_inv_Kzxxz.transpose(-2, -1), Kblk)

                Xcorr1 = (Kxz.transpose(-2, -1).matmul(- mq.matmul(Aq.transpose(-2, -1)) + bq) * wwLeg.unsqueeze(-1).unsqueeze(-1)).sum(0).sum(0)
                Xcorr2 = (dKxz.transpose(-2, -3).squeeze(-2).matmul(Sq).matmul(Aq.transpose(-2, -1)) * wwLeg.unsqueeze(-1).unsqueeze(-1)).sum(0).sum(0)

                Kblk_inv_Xcorr1, _ = torch.solve(Xcorr1, Kblk)
                Kblk_inv_Xcorr2, _ = torch.solve(Xcorr2, Kblk)
                # update for q_sigma (same across all dimensions)
                CovMat = Kzz_inv + Kblk_inv_Kzxxz_Kblk_inv[..., :self.numZ, :self.numZ]  # M x M
                self.q_sigma = (torch.inverse(CovMat.squeeze(0)).unsqueeze(0)).repeat(self.xDim, 1, 1)  # (Kzz_inv + [Kblk_inv Kzxxz Kblk_inv]_M )^{-1} repeated for each dimension K x M x M

                # jointly update q_mu and q_Jfx
                idx_mu = torch.arange(self.numZ)  # indices for inducing point
                idx_Jfx = torch.arange(self.numFx * self.xDim) + self.numZ + self.numFx  # indices for jacobians
                idx = torch.cat([idx_mu, idx_Jfx], dim=0)  # all indices
                # for slicing matrices
                CovMat_joint = KLmat[:, idx, :][:, :, idx] + Kblk_inv_Kzxxz_Kblk_inv[:, idx, :][:, :, idx]
                q_joint_new, _ = torch.solve(Kblk_inv_Xcorr1.squeeze(0)[idx, :] - Kblk_inv_Xcorr2.squeeze(0)[idx, :], CovMat_joint.squeeze(0))

                self.q_mu = q_joint_new[:self.numZ, :]
                self.q_Jfx = q_joint_new[-self.numFx * self.xDim:, :]
        else:
            pass

    def assemble_qs(self):
        q_u = self.q_mu  # M x K
        q_fx = torch.zeros_like(self.Zs_fx).type(float_type)  # F x K
        q_Jfx = self.q_Jfx  # FK x K
        q_mu = torch.cat([q_u, q_fx, q_Jfx], dim=0)

        if self.q_sigma.view(self.numZ, self.xDim, -1).size(-1) == 1:  # diagonal case
            q_diag_u = self.q_sigma
            q_diag_Jfx = torch.zeros_like(self.q_Jfx).type(float_type)  # FK x K
            q_diag = torch.cat([q_diag_u, q_fx, q_diag_Jfx], dim=0)
            q_sigma = batch_vec_to_diag(q_diag.transpose(-2, -1))  # K x M x M
        else:
            ndims = self.numFx + self.numZ + self.numFx * self.xDim
            q_sigma = torch.zeros(self.xDim, ndims, ndims).type(float_type)  # M + F + FK
            # put full covariance into first block, rest is zeros
            q_sigma[..., :self.numZ, :self.numZ] = self.q_sigma

        return q_mu, q_sigma

    def get_conditional_blocks(self):
        # get matrices needed for KL[q(u)|| p(u)]
        # Kblock is condtional Kzz
        # Kvec is [Kzs Kzd]
        # Kcond is [Kss Ksd; Kds Kdd]

        # get individual blocks to carefully penalise for the correct things
        Kzz = self.kern(mode="k", x1=self.Zs.unsqueeze(0), x2=self.Zs.unsqueeze(0))
        Kzz = Kzz + (self.jitter_level * torch.eye(self.numZ, device=Kzz.device).unsqueeze(0)).type(float_type)

        Kss = self.kern(mode="k", x1=self.Zs_fx.unsqueeze(0), x2=self.Zs_fx.unsqueeze(0)) + self.Zs_fx_sqrt.pow(2).view(-1).diag()

        Kss = Kss + (self.jitter_level * torch.eye(self.numFx, device=Kss.device).unsqueeze(0)).type(float_type)

        Kzs = self.kern(mode="k", x1=self.Zs.unsqueeze(0), x2=self.Zs_fx.unsqueeze(0))
        Kdz = self.kern(mode="d1k", x1=self.Zs_fx.unsqueeze(0), x2=self.Zs.unsqueeze(0))  # d1 k(s,z)
        Kds = self.kern(mode="d1k", x1=self.Zs_fx.unsqueeze(0), x2=self.Zs_fx.unsqueeze(0))  # d1 k(s,s)

        # reshape into expected format: R x F x (M + F) x K --- >  R x K F x (M+F) stacked as [d1K(fx1,z), d1K(fx2,z), d1K(fx3,z), ...]
        Kdz = stack_along_dim(Kdz, dim_unbind=-3, dim_stack=-1)
        Kds = stack_along_dim(Kds, dim_unbind=-3, dim_stack=-1)

        # get covariance for derivative kernel
        Kdd = self.kern(mode="d1d2k", x1=self.Zs_fx.unsqueeze(0), x2=self.Zs_fx.unsqueeze(0))  # R x M2 x M2 x K x K
        # reshape into expected format: R x F x F x K x K --- >  R x K F x (M1 F) stacked as
        # [d1d2K(s1,s1), d1d2K(s1,s2), ...]
        # [d1d2K(s2,s1), d1d2K(s2,s2), ...]. each entry is a K x K block
        # [   ... ,       ...  ,    ...   ]
        Kdd = stack_along_dim(stack_along_dim(Kdd, dim_unbind=-3, dim_stack=-1), dim_unbind=-3, dim_stack=-1)

        Kvec = torch.cat([Kzs, Kdz.transpose(-2, -1)], dim=-1)

        Kcond = torch.cat([torch.cat([Kss, Kds.transpose(-2, -1)], dim=-1), torch.cat([Kds, Kdd], dim=-1)], dim=-2)

        Kcond_inv_Kvec, _ = torch.solve(Kvec.transpose(-1, -2), Kcond)

        Kblock = Kzz - Kvec.matmul(Kcond_inv_Kvec)  # conditional prior covariance

        return Kblock, Kcond, Kvec

    def kullback_leibler(self):
        # Kullback leibler divergence between inducing points posterior and prior

        # q_mu = torch.cat([self.q_mu, self.q_Jfx], dim=0)
        # q_diag = torch.cat([self.q_diag, self.q_diag_Jfx], dim=0)
        q_mu = self.q_mu
        if self.q_sigma.view(self.numZ, self.xDim, -1).size(-1) == 1:  # diagonal case
            q_sigma = batch_vec_to_diag(self.q_diag.pow(2).transpose(-2, -1))
        else:
            q_sigma = self.q_sigma

        # means from prior
        q_fx = torch.zeros_like(self.Zs_fx)  # F x K
        q_cond = torch.cat([q_fx, self.q_Jfx], dim=0)

        numZ = q_mu.size(0)

        Kblock, Kcond, Kvec = self.get_conditional_blocks()

        # trace( Kzz_inv * q_sigma)
        Kblockinv_sig, _ = torch.solve(q_sigma.sum(0), Kblock)

        tt1 = batch_diag(Kblockinv_sig).sum()

        # (q_mu- p_mu)' * Kzz_inv * (q_mu - p_mu)
        Kcond_inv_q_cond, _ = torch.solve(q_cond, Kcond)

        p_mu = Kvec.matmul(Kcond_inv_q_cond)

        KBlockinv_q, _ = torch.solve(q_mu - p_mu, Kblock)

        tt2 = (KBlockinv_q * (q_mu - p_mu)).sum()

        # logdet(Kzz) - logdet(q_sigma) - k
        tt3 = self.xDim * apply_along_axis(logdet, Kblock, dim=0).sum()

        tt4 = - apply_along_axis(logdet, q_sigma, dim=0).sum() - self.xDim * numZ

        # sum everything up
        kld = torch.sum(0.5 * (tt1 + tt2 + tt3 + tt4))

        if torch.any(kld < 0):  # make sure KLD is non-zero
            warnings.warn('negative KL divergence in transition function encountered')

        return kld

    def concatenate_inputs(self):
        return torch.cat([self.Zs, self.Zs_fx], dim=0)

    def get_block_matrices(self):
        Zzx = self.concatenate_inputs()  # concatenate inputs for building standard kernels
        # get uncertainty in fixed points
        Zs_fx_uncertainty = torch.zeros(self.numFx + self.numZ).type(float_type).diag()
        Zs_fx_uncertainty[..., -self.numFx:, -self.numFx:] = self.Zs_fx_sqrt.pow(2).view(-1).diag()

        # add this to diagonal plus some jitter for stability
        Kzz = self.kern(mode="k", x1=Zzx, x2=Zzx) + Zs_fx_uncertainty
        Kzz = Kzz + (self.jitter_level * torch.eye(self.numZ + self.numFx, device=Kzz.device).unsqueeze(0)).type(float_type)  # R x (M1+M2) x (M1+M2)

        # get cross covariance between derivative and standard kernel
        Kdz = self.kern(mode="d1k", x1=self.Zs_fx.unsqueeze(0), x2=Zzx)

        # reshape into expected format: R x F x (M + F) x K --- >  R x K F x (M+F) stacked as [d1K(fx1,z), d1K(fx2,z), d1K(fx3,z), ...]
        Kdz = stack_along_dim(Kdz, dim_unbind=-3, dim_stack=-1)

        # get covariance for derivative kernel
        Kdd = self.kern(mode="d1d2k", x1=self.Zs_fx.unsqueeze(0), x2=self.Zs_fx.unsqueeze(0))  # R x M2 x M2 x K x K
        # reshape into expected format: R x F x F x K x K --- >  R x K F x (M1F) stacked as
        # [d1d2K(s1,s1), d1d2K(s1,s2), ...]
        # [d1d2K(s2,s1), d1d2K(s2,s2), ...]. each entry is a K x K block
        # [   ... ,       ...  ,    ...   ]
        Kdd = stack_along_dim(stack_along_dim(Kdd, dim_unbind=-3, dim_stack=-1), dim_unbind=-3, dim_stack=-1)

        return Kzz, Kdz, Kdd

    def get_Kzz(self):
        Kzz, Kdz, Kdd = self.get_block_matrices()
        # Kblock = torch.cat([torch.cat([Kzz, Kdz.transpose(-2, -1)], dim=2), torch.cat([Kdz, Kdd], dim=2)], dim=1)
        Kblock = torch.cat([torch.cat([Kzz, Kdz.transpose(-2, -1)], dim=-1), torch.cat([Kdz, Kdd], dim=-1)], dim=-2)

        return Kblock

    def get_Kzz_inv(self):
        # output is 1 x (M1+M2+M2*K) x (M1+M2+M2*K)
        Kzz, Kdz, Kdd = self.get_block_matrices()

        Kdd_inv_Kdz, _ = torch.solve(Kdz, Kdd)
        Kzz_inv_Kzd, _ = torch.solve(Kdz.transpose(-2, -1), Kzz)

        # use schur complement to compute block inverses  inverse = [[A, B], [C D]]
        block1 = batch_inverse(Kzz - Kdz.transpose(-2, -1).matmul(Kdd_inv_Kdz))  # block A of inverse

        block2 = batch_inverse(Kdd - Kdz.matmul(Kzz_inv_Kzd))  # block D of inverse

        block3 = -block2.matmul(Kzz_inv_Kzd.transpose(-2, -1))  # block C = B' of inverse

        Kblock_inv = torch.cat([torch.cat([block1, block3.transpose(-2, -1)], dim=2), torch.cat([block3, block2], dim=2)], dim=1)

        return Kblock_inv

    def get_Kxz(self, x):
        # assemble vector for predictions
        Zzx = self.concatenate_inputs()
        Kxz = self.kern(mode="k", x1=x, x2=Zzx)
        Kxd = self.kern(mode="d2k", x1=x, x2=self.Zs_fx.unsqueeze(0))  # R x T x F x K
        Kxd = stack_along_dim(Kxd, dim_unbind=-2, dim_stack=-1)

        return torch.cat([Kxz, Kxd], dim=-1)

    def get_E_Kxz(self, m, S):
        Zzx = self.concatenate_inputs()  # concatenate inputs for building standard kernels
        Kxz = self.kern(mode="psi1", x2=unsqueeze_as(Zzx, m), mu=m, cov=S)
        Kxd = self.kern(mode="psid2", x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S)  # R x 1 x F x K
        Kxd = stack_along_dim(Kxd, dim_unbind=-2, dim_stack=-1)  # R x 1 x FK
        return torch.cat([Kxz, Kxd], dim=-1)

    def get_dm_E_Kxz(self, m, S):
        Zzx = self.concatenate_inputs()
        Kxz = self.kern.dPsi1dmu(x2=unsqueeze_as(Zzx, m), mu=m, cov=S)  # (R x N1 x N2) x (1 x K)
        Kxd = self.kern.dPsid2dmu(x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S)  # (R x 1 x M x K) x (1 x K)
        # reshape into expected format
        Kxd = stack_along_dim(Kxd, dim_unbind=-4, dim_stack=-3).transpose(-1, -2)

        return torch.cat([Kxz, Kxd], dim=-3)

    def get_dS_E_Kxz(self, m, S):
        Zzx = self.concatenate_inputs()  # concatenate inputs for building standard kernels
        Kxz = self.kern.dPsi1dcov(x2=unsqueeze_as(Zzx, m), mu=m, cov=S)  # (R x N1 x N2) x (K x K)
        Kxd = self.kern.dPsid2dcov(x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S)  # (R x 1 x M x K) x (K x K)
        Kxd = stack_along_dim(Kxd, dim_unbind=-4, dim_stack=-3).transpose(-2, -1)

        return torch.cat([Kxz, Kxd], dim=-3)

    def get_E_dKxz(self, m, S):
        # R x 1 x M x K
        Zzx = self.concatenate_inputs()
        dKxz = self.kern(mode="psid1", x2=unsqueeze_as(Zzx, m), mu=m, cov=S)  # R x 1 x (M+F) x K
        dKxd = self.kern(mode="psid1d2", x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S)  # R x 1 x F x K x K
        dKxd = stack_along_dim(dKxd, dim_unbind=-3, dim_stack=-1)  # R x 1 x FK x K
        return torch.cat([dKxz, dKxd], dim=-2)

    def get_dm_E_dKxz(self, m, S):
        # (R x 1 x M x K) x (1 x K)
        Zzx = self.concatenate_inputs()
        dKxz = self.kern.dPsid1dmu(x2=unsqueeze_as(Zzx, m), mu=m, cov=S)  # R x 1 x (M+F) x K x 1 x K
        dKxd = self.kern.dPsid1d2dmu(x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S)  # R x 1 x F x K x K x 1 x K
        dKxd = stack_along_dim(dKxd, dim_unbind=-5, dim_stack=-3).transpose(-1, -2)
        return torch.cat([dKxz, dKxd], dim=-4)

    def get_dS_E_dKxz(self, m, S):
        # (R x 1 x M x K) x (K x K)
        Zzx = self.concatenate_inputs()
        dKxz = self.kern.dPsid1dcov(x2=unsqueeze_as(Zzx, m), mu=m, cov=S)  # R x 1 x (M+F) x K x k x K
        dKxd = self.kern.dPsid1d2dcov(x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S)  # R x 1 x F x K x K x K x K
        dKxd = stack_along_dim(dKxd, dim_unbind=-5, dim_stack=-3).transpose(-2, -1)
        return torch.cat([dKxz, dKxd], dim=-4)

    def get_E_Kzxxz(self, m, S):
        # get blocks of matrix

        Zzx = self.concatenate_inputs()
        Kzsxz = self.kern(mode="psi2", x2=unsqueeze_as(Zzx, m), mu=m, cov=S)  # R x M+F x M+F
        # Kzsxz = Kzsxz + (self.jitter_level**2 * torch.eye(self.numFx + self.numZ, device=Kzsxz.device).unsqueeze(0))

        dKsxxz = self.kern(mode="psid1psi1", x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S, x3=unsqueeze_as(Zzx, m))  # R x F x M+F x K
        ddKsxxs = self.kern(mode="psid1psid2", x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S)  # R x FK x FK

        # reshape into expected format: R x F x F x K x K --- >  R x K F x (KF) stacked as
        # [d1K(s1,s1)d2K(s1,s1), d1K(s1,s2)d2K(s1,s2), ...]
        # [d1K(s2,s1)d2K(s2,s1), d1K(s2,s2)d2K(s2,s2), ...]. each entry is a K x K block
        # [   ... ,       ...  ,    ...   ]

        # ddKsxxs1 = torch.cat(torch.unbind(torch.cat(torch.unbind(ddKsxxs, dim=-4), dim=-2), dim=-3), dim=-1)  # R x FK x FK
        ddKsxxs = stack_along_dim(stack_along_dim(ddKsxxs, dim_unbind=-3, dim_stack=-1), dim_unbind=-3, dim_stack=-1)

        # reshape into expected format: R x F x M x K --- >  R x KF x M stacked as [d1K(fx1,z), d1K(fx2,z), d1K(fx3,z), ...]
        dKsxxz = stack_along_dim(dKsxxz, dim_unbind=-3, dim_stack=-1)

        # assemble blocks into full matrix
        row1 = torch.cat([Kzsxz, dKsxxz.transpose(-2, -1)], dim=-1)
        row2 = torch.cat([dKsxxz, ddKsxxs], dim=-1)
        return torch.cat([row1, row2], dim=-2)

    def get_dm_E_Kzxxz(self, m, S):
        # (R x M x M) x 1 x K
        Zzx = self.concatenate_inputs()
        Kzsxz = self.kern.dPsi2dmu(x2=unsqueeze_as(Zzx, m), mu=m, cov=S)  # R x M+F x M+F x 1 x K
        dKsxxz = self.kern.dPsid1Psi1dmu(x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S, x3=unsqueeze_as(Zzx, m))  # R x F x M+F x K x 1 x K
        ddKsxxs = self.kern.dPsid1Psid2dmu(x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S)  # R x F x F x K x K x 1 x K
        # reshape into expected format: R x F x F x K x K --- >  R x K F x (KF) stacked as
        # [d1K(s1,s1)d2K(s1,s1), d1K(s1,s2)d2K(s1,s2), ...]
        # [d1K(s2,s1)d2K(s2,s1), d1K(s2,s2)d2K(s2,s2), ...]. each entry is a K x K block
        # [   ... ,       ...  ,    ...   ]
        ddKsxxs = stack_along_dim(stack_along_dim(ddKsxxs, dim_unbind=-5, dim_stack=-3), dim_unbind=-5, dim_stack=-3)

        # reshape into expected format: R x F x M  x K --- >  R x KF x M stacked as [d1K(fx1,z), d1K(fx2,z), d1K(fx3,z), ...]
        dKsxxz = stack_along_dim(dKsxxz, dim_unbind=-5, dim_stack=-3).transpose(-1, -2)

        # assemble blocks into full matrix
        row1 = torch.cat([Kzsxz, dKsxxz.transpose(-4, -3)], dim=-3)
        row2 = torch.cat([dKsxxz, ddKsxxs], dim=-3)
        return torch.cat([row1, row2], dim=-4)

    def get_dS_E_Kzxxz(self, m, S):
        # (R x M x M) x K x K
        Zzx = self.concatenate_inputs()

        Kzsxz = self.kern.dPsi2dcov(x2=unsqueeze_as(Zzx, m), mu=m, cov=S)
        dKsxxz = self.kern.dPsid1Psi1dcov(x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S, x3=unsqueeze_as(Zzx, m))
        ddKsxxs = self.kern.dPsid1Psid2dcov(x2=unsqueeze_as(self.Zs_fx, m), mu=m, cov=S)

        # reshape into expected format: R x F x F x K x K --- >  R x K F x (KF) stacked as
        # [d1K(s1,s1)d2K(s1,s1), d1K(s1,s2)d2K(s1,s2), ...]
        # [d1K(s2,s1)d2K(s2,s1), d1K(s2,s2)d2K(s2,s2), ...]. each entry is a K x K block
        # [   ... ,       ...  ,    ...   ]
        ddKsxxs = stack_along_dim(stack_along_dim(ddKsxxs, dim_unbind=-5, dim_stack=-3), dim_unbind=-5, dim_stack=-3)

        # reshape into expected format: R x F x M  x K --- >  R x KF x M stacked as [d1K(fx1,z), d1K(fx2,z), d1K(fx3,z), ...]
        dKsxxz = stack_along_dim(dKsxxz, dim_unbind=-5, dim_stack=-3).transpose(-1, -2)

        # assemble blocks into full matrix
        row1 = torch.cat([Kzsxz, dKsxxz.transpose(-4, -3)], dim=-3)
        row2 = torch.cat([dKsxxz, ddKsxxs], dim=-3)
        return torch.cat([row1, row2], dim=-4)
