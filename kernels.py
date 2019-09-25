import torch
from torch import nn
from utils import unsqueeze_as, batch_inverse_psd, batch_make_diag, batch_det_psd
from settings import float_type


class StationaryKernel(nn.Module):
    def __init__(self):
        super(StationaryKernel, self).__init__()

        self.output_scale = nn.Parameter(torch.ones(1).type(float_type))

    def paired_diff(self, x1, x2):
        """
        returns pairwise differences of
        X1 = M x R x N1 x D and
        X2 = M x R x N2 x D
        as
        diff = M x R x N1 x N2 x D
        """
        # diff = x1.unsqueeze(2) - x2.unsqueeze(1)
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)

        return diff

    def scaled_paired_diff(self, x1, x2, cov):
        """
        returns pairwise scaled differences of
        X1 = (V x ) x R x N1 x D and
        X2 = (V x ) x R x N2 x D
        cov = (V x ) x R x D x D
        as
        diff = (V x ) R x N1 x N2 x D =(x1 - x2) * inv(cov)
        also possible to add extra batch dim
        """
        diff = self.paired_diff(x1, x2)  # (x1 - x2) (V x ) R x N1 x N2 x D

        # cov_inv = apply_along_batch_axis(torch.inverse, cov)
        # res1 = diff.matmul(cov_inv.unsqueeze(-3))
        res1, _ = torch.solve(diff.transpose(-2, -1), cov.unsqueeze(-3))
        return res1.transpose(-2, -1)

    def squared_mahalanobis_distance(self, x1, x2, cov):
        """
        returns pairwise scaled distances of
        X1 = R x N1 x D and
        X2 = R x N2 x D
        cov = R x D x D
        as
        dist = R x N1 x N2
        """

        diff = self.paired_diff(x1, x2)  # (x1 - x2)
        diff_scaled = self.scaled_paired_diff(x1, x2, cov)  # R x N1 x N2 x D
        return (diff * diff_scaled).sum(-1)  # R x N1 x N2

    def Kdiag(self, x1):
        """
        diagonal of kernel function
        """
        return self.output_scale**2 * torch.ones(list(x1.size())[:-1]).type(float_type)

    def K(self, x1, x2):
        """
        classic kernel function
        """
        raise NotImplementedError()

    def d1K(self, x1, x2):
        """
        derivative kernel function wrt first input argument
        """
        raise NotImplementedError()

    def d1d2K(self, x1, x2):
        """
        derivative kernel function wrt first input and second input argument
        """
        raise NotImplementedError()

    def Psi0(self, mu, cov):
        """
        expectation of diagonal kernel function
        mu is R x 1 x K
        output is R x 1
        """
        return self.output_scale**2 * torch.ones(list(mu.size())[:-1]).type(float_type)

    def dPsi0dmu(self, mu, cov):
        """
        gradient with respect to mean of input: R x 1 x 1 x K
        """
        return torch.zeros_like(mu).unsqueeze(1)

    def dPsi0dcov(self, mu, cov):
        """
        gradient with respect to covariance of input: R x 1 x K x K
        """
        return torch.zeros_like(cov).unsqueeze(1)

    def Psi1(self, x2, mu, cov):
        """
        expectation of classic kernel function computed using Gauss-Hermite quadrature
        """
        raise NotImplementedError()

    def dPsi1dmu(self, mu, cov):
        raise NotImplementedError()

    def dPsi1dcov(self, mu, cov):
        raise NotImplementedError()

    def Psid1(self, x2, mu, cov):
        """
        expectation of derivative kernel function computed using Gauss-Hermite quadrature
        """
        raise NotImplementedError()

    def dPsid1dmu(self, mu, cov):

        raise NotImplementedError()

    def dPsid1dcov(self, mu, cov):

        raise NotImplementedError()

    def Psi2(self, x2, mu, cov):
        """
        expectation of product of classic kernel function <k(x,x2)k(x2,x)> computed using Gauss-Hermite quadrature
        """
        raise NotImplementedError()

    def dPsi2dmu(self, mu, cov):
        raise NotImplementedError()

    def dPsi2dcov(self, mu, cov):
        raise NotImplementedError()

    def forward(self, mode, **kwargs):

        if mode == "k":
            # classic kernel
            K = self.K(kwargs['x1'], kwargs['x2'])
        elif mode == "kdiag":
            # diagonal kernel evaluated at same inputs
            K = self.Kdiag(kwargs['x1'])
        elif mode == "psi0":
            # <k(x,x)>
            K = self.Psi0(kwargs['mu'], kwargs['cov'])
        elif mode == "psi1":
            # <k(x,z)>
            K = self.Psi1(kwargs['x2'], kwargs['mu'], kwargs['cov'])
        elif mode == "psi2":
            # <k(x,z)k(z,x)'>
            K = self.Psi2(**kwargs)
        elif mode == "d1k":
            # d/dxk(x,z)
            K = self.d1K(kwargs['x1'], kwargs['x2'])
        elif mode == "d2k":
            # d/dxk(x,z)
            K = -self.d1K(kwargs['x1'], kwargs['x2'])
        elif mode == "d1d2k":
            # d/dxk(x,z) d/dx k(z,x)
            K = self.d1d2K(kwargs['x1'], kwargs['x2'])
        elif mode == "psid1":
            # <d/dxk(x,z)>
            K = self.Psid1(kwargs['x2'], kwargs['mu'], kwargs['cov'])
        elif mode == "psid2":
            # <d/dxk(x,z)>
            K = self.Psid2(kwargs['x2'], kwargs['mu'], kwargs['cov'])
        elif mode == "psid1psid2":
            # <d/dxk(x,z) d/dxk(z,x)>
            K = self.Psid1Psid2(**kwargs)
        elif mode == "psid1psi1":
            # <d/dx k(x,z) k(z,x) >
            K = self.Psid1Psi1(**kwargs)
        elif mode == "psid1d2":
            # <d/dx k(x,z) k(z,x) >
            K = self.Psid1d2(kwargs['x2'], kwargs['mu'], kwargs['cov'])
        else:
            raise("Unknown kernel mode specified")

        return K


class RBF(StationaryKernel):
    def __init__(self, num_dims, lengthscales_Init=None):
        super(RBF, self).__init__()
        if lengthscales_Init is None:
            self.lengthscales = nn.Parameter(torch.ones(num_dims, 1).type(float_type))  # D x 1
        else:
            self.lengthscales = nn.Parameter(lengthscales_Init.view(num_dims, 1).type(float_type))  # D x 1
        # self.lengthscales = nn.Parameter(torch.rand(num_dims, 1) + 0.5)  # D x 1
        """
        Exponentiated Quadratic kernel class.
        The forward call needs to provide keyworded input arguments:
        mode = {k, kdiag, psi0, psi1, psi2}
        x1 = first input
        x2 = second input
        mu = mean for psi statistics
        cov = covariance for psi statistics
        """

    def K(self, x1, x2):
        """
        classic kernel function
        """
        diff = self.paired_diff(x1, x2)

        diffsq = torch.sum(torch.div(diff, self.lengthscales.view(1, 1, 1, -1))**2, dim=3)

        return self.output_scale**2 * torch.exp(-0.5 * diffsq)

    def d1K(self, x1, x2):
        """
        derivative of kernel function with respect to the first input argument
        Output is (R x N1 x N2) x K
        """
        diff = self.paired_diff(x1, x2)

        scaled_diff = torch.div(diff, self.lengthscales.pow(2).view(1, -1))

        return - self.K(x1, x2).unsqueeze(-1) * scaled_diff

    def d1d2K(self, x1, x2):
        """
        derivative of kernel function with respect to the first input argument
        Output is (R x N1 x N2) x (K x K)
        """
        diff = self.paired_diff(x1, x2)  # R x N1 x N2 x K

        scaled_diff = torch.div(diff, self.lengthscales.pow(2).view(1, -1))  # A\inv (x - x')

        outer = scaled_diff.unsqueeze(-1) * scaled_diff.unsqueeze(-2)  # R x N1 x N2 x K x K  A\inv (x - x') (x - x')' A\inv

        return ((1. / self.lengthscales.pow(2).squeeze(1)).diag() - outer) * self.K(x1, x2).unsqueeze(-1).unsqueeze(-1)

    def Psi1(self, x2, mu, cov):
        """
        expectation of classic kernel function

        x2 is R x N1 x K
        mu is R x N2 x K
        cov is R x K x K

        Output is R x N1 x N2
        """
        lengthscales_new_squared = self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # R x K x K  is (A + S)

        diffsq = self.squared_mahalanobis_distance(mu, x2, lengthscales_new_squared)  # is (m - z)*inv(A + S)*(m - z)

        output_variance_new = self.output_scale**2 * \
            batch_det_psd(cov.div(self.lengthscales.pow(2).view(1, -1)) +
                          torch.eye(self.lengthscales.size(0), device=x2.device).type(float_type)).pow(-0.5)

        return unsqueeze_as(output_variance_new, diffsq, dim=-1) * diffsq.mul(-0.5).exp()  # R x N1 x N2

    def Psi2(self, x2, mu, cov, x3=None):
        """
        order expectation of product of classic kernel function <k(x,x2)k(x2,x)>

        x2 is R x M x K
        mu is R x 1 x K
        cov is R x K x K

        Output is R x M x M

        """
        if mu.size(-2) != 1:  # Psi2 not working for matrix version
            raise('dimensionality mismatch: input mean needs to be a R x 1 x K vector')

        if x3 is None:
            x3 = x2  # we actually want this to point to the same object

        x2_diff_sq = self.squared_mahalanobis_distance(x2, x3, self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0))  # (z-z')A^{-1}(z-z')

        lengthscales_new_squared = 0.5 * self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # ( 1/2 A + S)

        x2_new = 0.5 * (x2.unsqueeze(-2) + x3.unsqueeze(-3))  # R x M x M x K is (z + z')/2

        mu_x2_new_diffs = mu.unsqueeze(-2) - x2_new  # R x M x M x K is (m - (z + z') / 2 )

        mu_x2_new_diff_scaled, _ = torch.solve(mu_x2_new_diffs.transpose(-1, -2), lengthscales_new_squared.unsqueeze(-3))

        mu_x2_new_diff_sq = (mu_x2_new_diff_scaled.transpose(-1, -2) * mu_x2_new_diffs).sum(-1)  # (m - (z + z') / 2 )' * inv( 1/2 A + S) (m - (z + z') / 2 )

        output_variance_new = self.output_scale**4 * batch_det_psd(2 * cov.div(self.lengthscales.pow(2).view(1, -1)) +
                                                                   torch.eye(self.lengthscales.size(0), device=x2.device).type(float_type)).pow(-0.5)

        return output_variance_new * (- 0.5 * mu_x2_new_diff_sq - 0.25 * x2_diff_sq).exp()

    def dPsi1dmu(self, x2, mu, cov):
        """
        derivative of Psi1 wrt mu
        Output is (R x N1 x N2) x (1 x K)
        """
        lengthscales_new_squared = self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # (A + S)

        scaled_diffs = self.scaled_paired_diff(mu, x2, lengthscales_new_squared)

        return - (self.Psi1(x2, mu, cov).unsqueeze(-1) * scaled_diffs).unsqueeze(-2)  # - Psi * (A + S)^{-1} (m - z)

    def dPsi2dmu(self, x2, mu, cov, x3=None):
        """
        derivative of Psi2 wrt mu
        Output is (R x M x M) x (1 x K)
        """
        if mu.size(-2) != 1:  # Psi2 not working for matrix version
            raise('dimensionality mismatch: input mean needs to be a R x 1 x K vector')

        if x3 is None:
            x3 = x2  # we actually want this to point to the same object

        lengthscales_new_squared = 0.5 * self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov

        x2_new = 0.5 * (x2.unsqueeze(2) + x3.unsqueeze(1))  # (z + z')/2

        mu_x2_new_diffs = mu.unsqueeze(1) - x2_new  # R x M x M x K is (m - (z + z') / 2 )

        scaled_diffs, _ = torch.solve(mu_x2_new_diffs.transpose(-1, -2), lengthscales_new_squared.unsqueeze(-3))

        return - (self.Psi2(x2, mu, cov, x3).unsqueeze(-1) * scaled_diffs.transpose(-1, -2)).unsqueeze(-2)

    def dPsi1dcov(self, x2, mu, cov):
        """
        derivative of Psi1 wrt mu
        Output is (R x 1 x M) x (K x K)
        """
        lengthscales_new_squared = self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # (A + S)

        scaled_diffs = self.scaled_paired_diff(mu, x2, lengthscales_new_squared)  # (A + S)^{-1} (m - z)

        term1 = scaled_diffs.unsqueeze(-1) * scaled_diffs.unsqueeze(-2)  # (m - z)'(A + S)^{-1} dS/dS_{ij} (A + S)^{-1} (m - z)

        term2 = batch_inverse_psd(lengthscales_new_squared)  # A^{-1} (S A^{-1} + I)^{-1}  = (A + S)^{-1} R x K x K

        dcov = 0.5 * self.Psi1(x2, mu, cov).unsqueeze(-1).unsqueeze(-1) * (term1 - term2.unsqueeze(1).unsqueeze(1))

        return dcov + dcov.transpose(-2, -1) - batch_make_diag(dcov)  # (R x N1 x N2) x (K x K)

    def dPsi2dcov(self, x2, mu, cov, x3=None):
        if x3 is None:
            x3 = x2  # we actually want this to point to the same object

        lengthscales_new_squared = 0.5 * self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # (1/2 A + S)

        x2_new = 0.5 * (x2.unsqueeze(2) + x3.unsqueeze(1))  # (z + z') / 2

        mu_x2_new_diffs = mu.unsqueeze(1) - x2_new  # R x M x M x K is (m - (z + z') / 2 )

        fromDet = 2 * cov.matmul((1. / self.lengthscales.squeeze(1).pow(2)).diag()) + torch.eye(self.lengthscales.size(0), device=x2.device).type(float_type)  # (2 S A^{-1} + I)

        scaled_diffs, _ = torch.solve(mu_x2_new_diffs.transpose(-1, -2), lengthscales_new_squared.unsqueeze(-3))

        term1 = scaled_diffs.transpose(-1, -2).unsqueeze(-1) * scaled_diffs.transpose(-1, -2).unsqueeze(-2)  # (m - (z + z') / 2 )'(1/2A + S)^{-1} dS/dS_{ij} (1/2A + S)^{-1} (m - (z + z') / 2 )

        term2 = batch_inverse_psd(fromDet).div(self.lengthscales.pow(2))  # A^{-1} (2 S A^{-1} + I)^{-1}

        dcov = self.Psi2(x2, mu, cov, x3).unsqueeze(-1).unsqueeze(-1) * (0.5 * term1 - term2.unsqueeze(1).unsqueeze(1))  # (R x M x M) x (K x K)

        return dcov + dcov.transpose(-2, -1) - batch_make_diag(dcov)

    def Psid1(self, x2, mu, cov):
        """
        expectation of derivative of kernel function
        output is (R x N1 x N2) x K
        """
        new_cov = self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # (S + A)

        scaled_diff = self.scaled_paired_diff(mu, x2, new_cov)  # (S + A)^{-1}(m - z)   is R x 1 x N2 x K

        return -self.Psi1(x2, mu, cov).unsqueeze(-1) * scaled_diff  # R x N1 x N2 x K

    def dPsid1dmu(self, x2, mu, cov):
        """
        derivative of Psid1 with respect to mu
        """
        if mu.size(-2) != 1:  # Psi2 not working for matrix version
            raise('dimensionality mismatch: input mean needs to be a R x 1 x K vector')

        new_cov = self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # (S + A)

        new_cov_inv = batch_inverse_psd(new_cov)

        scaled_diff = self.scaled_paired_diff(mu, x2, new_cov)  # (S + A)^{-1}(m - z)   is R x 1 x N2 x K

        term1 = - self.dPsi1dmu(x2, mu, cov).unsqueeze(-3) * scaled_diff.unsqueeze(-1).unsqueeze(-1)

        term2 = - new_cov_inv.unsqueeze(-1).unsqueeze(-1).permute(0, -1, 1, -2, 2).unsqueeze(1) * self.Psi1(x2, mu, cov).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return term1 + term2

    def dPsid1dcov(self, x2, mu, cov):
        """
        derivative of Psid1 with respect to cov
        """

        new_cov = self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # (S + A) # R x K x K

        new_cov_inv = batch_inverse_psd(new_cov)

        scaled_diff = self.scaled_paired_diff(mu, x2, new_cov)  # (S + A)^{-1}(m - z)   is R x 1 x N2 x K

        term0 = new_cov_inv.unsqueeze(1).unsqueeze(1).unsqueeze(-1) * (scaled_diff.unsqueeze(-2).unsqueeze(-2))

        term0_diag = batch_make_diag(term0)

        term1 = self.Psi1(x2, mu, cov).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (term0 + term0.transpose(-2, -1) - term0_diag)  # correcting for symmetric cov

        term2 = -self.dPsi1dcov(x2, mu, cov).unsqueeze(-3) * scaled_diff.unsqueeze(-1).unsqueeze(-1)  # (R x 1 x N2 x K) x (K x K)

        return term1 + term2

    def Psid2(self, x2, mu, cov):
        """
        E_x [d/ds k(x,s)]
        """
        return -self.Psid1(x2, mu, cov)

    def dPsid2dmu(self, x2, mu, cov):
        """
        d/dm E_x [d/ds k(x,s)]
        """
        return -self.dPsid1dmu(x2, mu, cov)

    def dPsid2dcov(self, x2, mu, cov):
        """
        d/dS E_x [d/ds k(x,s)]
        """
        return -self.dPsid1dcov(x2, mu, cov)

    def Psid1Psi1(self, x2, mu, cov, x3=None):
        """
        expectation: < d1k(x2, x) k(x, x3) >_q(x)~N(mu,cov)
        if x3 isn't supplied we use x3 = x2
        note x2 is the variable in the position where the derivative is taken
        """
        if mu.size(-2) != 1:  # Psi2 not working for matrix version
            raise('dimensionality mismatch: input mean needs to be a R x 1 x K vector')

        if x3 is None:
            x3 = x2  # we actually want this to point to the same object

        new_cov = 0.5 * self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # ( 1/2 A + S)

        x2_scaled = x2.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1} R x 1 x M2 x K

        x2_x3_sum = 0.5 * (x2.unsqueeze(-2) + x3.unsqueeze(-3))  # R x M2 x M3 x K is (s + z)/2

        x2_x3_sum_scaled = x2_x3_sum.matmul(cov.unsqueeze(-3).div(self.lengthscales.pow(2).view(-1, 1)))  # (S A^{-1} (s + z) / 2)' R x M2 x M3 x K

        mu_x2_new_diffs = 0.5 * mu.unsqueeze(-2) + x2_x3_sum_scaled  # R x M2 x M3 x K is (1/2 m + S A^{-1} (s + z) / 2 )

        scaled_diffs, _ = torch.solve(mu_x2_new_diffs.unsqueeze(-1), new_cov.unsqueeze(-3).unsqueeze(-3))  # R x M2 x M3 x K x 1

        # (1/2 m + S A^{-1} (z + z') / 2 )'* inv( 1/2 A + S)
        return self.Psi2(x2, mu, cov, x3).unsqueeze(-1) * (scaled_diffs.squeeze(-1) - x2_scaled.unsqueeze(-2))  # R x N2 x N3 x K

    def dPsid1Psi1dmu(self, x2, mu, cov, x3=None):

        if mu.size(-2) != 1:  # Psi2 not working for matrix version
            raise('dimensionality mismatch: input mean needs to be a R x 1 x K vector')

        if x3 is None:
            x3 = x2  # we actually want this to point to the same object

        new_cov = 0.5 * self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # ( 1/2 A + S)

        x2_scaled = x2.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1}

        x2_x3_sum = 0.5 * (x2.unsqueeze(-2) + x3.unsqueeze(-3))  # R x M2 x M3 x K is (s + z)/2

        x2_x3_sum_scaled = x2_x3_sum.matmul(cov.unsqueeze(-3).div(self.lengthscales.pow(2).view(-1, 1)))  # (S A^{-1} (s + z) / 2)' R x M2 x M3 x K

        mu_x2_new_diffs = 0.5 * mu.unsqueeze(-2) + x2_x3_sum_scaled  # R x M2 x M3 x K is (1/2 m + S A^{-1} (s + z) / 2 )

        scaled_diffs, _ = torch.solve(mu_x2_new_diffs.unsqueeze(-1), new_cov.unsqueeze(-3).unsqueeze(-3))  # R x M2 x M3 x K x 1

        new_cov_inv = batch_inverse_psd(new_cov)

        term1 = self.dPsi2dmu(x2, mu, cov, x3).unsqueeze(-3) * (scaled_diffs.squeeze(-1) - x2_scaled.unsqueeze(-2)).unsqueeze(-1).unsqueeze(-1)  # *(R x N2 x N3 x K) x (1 x K)

        term2 = 0.5 * new_cov_inv.unsqueeze(-1).unsqueeze(-1).permute(0, -1, 1, -2, 2).unsqueeze(1) * self.Psi2(x2, mu, cov, x3).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return term1 + term2

    def dPsid1Psi1dcov(self, x2, mu, cov, x3=None):

        if mu.size(-2) != 1:  # Psi2 not working for matrix version
            raise('dimensionality mismatch: input mean needs to be a R x 1 x K vector')

        if x3 is None:
            x3 = x2  # we actually want this to point to the same object

        new_cov = 0.5 * self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # ( 1/2 A + S)

        x2_scaled = x2.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1}

        x2_x3_sum = 0.5 * (x2.unsqueeze(-2) + x3.unsqueeze(-3))  # R x M2 x M3 x K is (s + z)/2

        x2_x3_sum_scaled = x2_x3_sum.matmul(cov.unsqueeze(-3).div(self.lengthscales.pow(2).view(-1, 1)))  # (S A^{-1} (s + z) / 2)' R x M2 x M3 x K

        mu_x2_new_diffs = 0.5 * mu.unsqueeze(-2) + x2_x3_sum_scaled  # R x M2 x M3 x K is (1/2 m + S A^{-1} (s + z) / 2 )

        scaled_diffs, _ = torch.solve(mu_x2_new_diffs.unsqueeze(-1), new_cov.unsqueeze(-3).unsqueeze(-3))  # R x M2 x M3 x K x 1

        new_cov_inv = batch_inverse_psd(new_cov)  # inv( 1/2 A + S)

        term0 = - new_cov_inv.unsqueeze(1).unsqueeze(1).unsqueeze(-1) * (scaled_diffs.transpose(-2, -1).unsqueeze(-3) -
                                                                         x2_x3_sum.div(self.lengthscales.pow(2).view(1, -1)).unsqueeze(-2).unsqueeze(-2))
        term0_diag = batch_make_diag(term0)

        term1 = self.Psi2(x2, mu, cov, x3).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (term0 + term0.transpose(-2, -1) - term0_diag)  # correcting for symmetric cov

        term2 = self.dPsi2dcov(x2, mu, cov, x3).unsqueeze(-3) * (scaled_diffs.squeeze(-1) - x2_scaled.unsqueeze(-2)).unsqueeze(-1).unsqueeze(-1)  # *(R x N2 x N3 x K) x (K x K)

        return term1 + term2

    def Psid1Psid2(self, x2, mu, cov, x3=None):
        """
        expectation: < d1k(x2, x) d2k(x, x3)>_q(x)~N(mu,cov)
        if x3 isn't supplied we use x3 = x2
        """
        if mu.size(-2) != 1:  # Psi2 not working for matrix version
            raise('dimensionality mismatch: input mean needs to be a R x 1 x K vector')

        if x3 is None:
            x3 = x2  # we actually want this to point to the same object

        new_cov = 0.5 * self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # ( 1/2 A + S)

        x2_scaled = x2.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1}
        x3_scaled = x3.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1}

        x2_x3_sum = 0.5 * (x2.unsqueeze(-2) + x3.unsqueeze(-3))  # R x M2 x M3 x K is (s + z)/2

        x2_x3_sum_scaled = x2_x3_sum.matmul(cov.div(self.lengthscales.pow(2).view(-1, 1)).unsqueeze(-3))  # (S A^{-1} (s + z) / 2)' R x M2 x M3 x K

        mu_x2_new_diffs = 0.5 * mu.unsqueeze(-2) + x2_x3_sum_scaled  # R x M2 x M3 x K is (1/2 m + S A^{-1} (s + z) / 2 )

        new_cov_inv = batch_inverse_psd(new_cov)  # inv( 1/2 A + S)

        scaled_diffs, _ = torch.solve(mu_x2_new_diffs.unsqueeze(-1), new_cov.unsqueeze(-3).unsqueeze(-3))  # R x M2 x M3 x K x 1

        meanx2 = scaled_diffs - x2_scaled.unsqueeze(-2).unsqueeze(-1)  # # ... x R x M2 x M3 x K x 1
        meanx3 = scaled_diffs - x3_scaled.unsqueeze(-3).unsqueeze(-1)  # # ... x R x M2 x M3 x K x 1

        Sigma = 0.5 * new_cov_inv.matmul(cov).div(self.lengthscales.pow(2).view(1, -1))  # R x K x K

        return self.Psi2(x2, mu, cov, x3).unsqueeze(-1).unsqueeze(-1) * \
            (Sigma.unsqueeze(-3).unsqueeze(-3) + meanx2 * meanx3.transpose(-2, -1))  # R x M2 x M3 x K x K

    def dPsid1Psid2dmu(self, x2, mu, cov, x3=None):
        # output is: R x F x F x (K x K) x (1 x K)
        if mu.size(-2) != 1:  # Psi2 not working for matrix version
            raise('dimensionality mismatch: input mean needs to be a R x 1 x K vector')

        if x3 is None:
            x3 = x2  # we actually want this to point to the same object

        new_cov = 0.5 * self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # ( 1/2 A + S)

        x2_scaled = x2.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1}
        x3_scaled = x3.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1}

        x2_x3_sum = 0.5 * (x2.unsqueeze(-2) + x3.unsqueeze(-3))  # R x M2 x M3 x K is (s + z)/2

        x2_x3_sum_scaled = x2_x3_sum.matmul(cov.div(self.lengthscales.pow(2).view(-1, 1)).unsqueeze(-3))  # (S A^{-1} (s + z) / 2)' R x M2 x M3 x K

        mu_x2_new_diffs = 0.5 * mu.unsqueeze(-2) + x2_x3_sum_scaled  # R x M2 x M3 x K is (1/2 m + S A^{-1} (s + z) / 2 )

        new_cov_inv = batch_inverse_psd(new_cov)  # inv( 1/2 A + S)

        scaled_diffs, _ = torch.solve(mu_x2_new_diffs.unsqueeze(-1), new_cov.unsqueeze(-3).unsqueeze(-3))  # R x M2 x M3 x K x 1

        meanx2 = scaled_diffs - x2_scaled.unsqueeze(-2).unsqueeze(-1)  # ... x R x M2 x M3 x K x 1
        meanx3 = scaled_diffs - x3_scaled.unsqueeze(-3).unsqueeze(-1)  # ... x R x M2 x M3 x K x 1

        Sigma = 0.5 * new_cov_inv.matmul(cov).div(self.lengthscales.pow(2).view(1, -1))  # R x K x K

        dmeandm = new_cov_inv.unsqueeze(-1).unsqueeze(-1).permute(0, -1, 1, -2, 2).unsqueeze(1).unsqueeze(1)

        term1 = self.dPsi2dmu(x2, mu, cov, x3).unsqueeze(-3).unsqueeze(-3) * (Sigma.unsqueeze(-3).unsqueeze(-3) +
                                                                              meanx2 * meanx3.transpose(-2, -1)).unsqueeze(-1).unsqueeze(-1)

        term2 = 0.5 * self.Psi2(x2, mu, cov, x3).unsqueeze(-1).unsqueeze(-1).unsqueeze(-3).unsqueeze(-3) * \
            (meanx2.unsqueeze(-1).unsqueeze(-1) * dmeandm + meanx3.transpose(-2, -1).unsqueeze(-1).unsqueeze(-1) * dmeandm.transpose(-3, -4))

        return term1 + term2

    def dPsid1Psid2dcov(self, x2, mu, cov, x3=None):
        """
        derivative of <d1kd2k> wrt covariance
        """
        if mu.size(-2) != 1:  # Psi2 not working for matrix version
            raise('dimensionality mismatch: input mean needs to be a R x 1 x K vector')

        if x3 is None:
            x3 = x2  # we actually want this to point to the same object

        new_cov = 0.5 * self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # ( 1/2 A + S)

        x2_scaled = x2.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1}
        x3_scaled = x3.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1}

        x2_x3_sum = 0.5 * (x2.unsqueeze(-2) + x3.unsqueeze(-3))  # R x M2 x M3 x K is (s + z)/2

        x2_x3_sum_scaled = x2_x3_sum.matmul(cov.div(self.lengthscales.pow(2).view(-1, 1)).unsqueeze(-3))  # (S A^{-1} (s + z) / 2)' R x M2 x M3 x K

        mu_x2_new_diffs = 0.5 * mu.unsqueeze(-2) + x2_x3_sum_scaled  # R x M2 x M3 x K is (1/2 m + S A^{-1} (s + z) / 2 )

        new_cov_inv = batch_inverse_psd(new_cov)  # inv( 1/2 A + S)

        scaled_diffs, _ = torch.solve(mu_x2_new_diffs.transpose(-1, -2), new_cov.unsqueeze(-3))

        scaled_diffs, _ = torch.solve(mu_x2_new_diffs.unsqueeze(-1), new_cov.unsqueeze(-3).unsqueeze(-3))  # R x M2 x M3 x K x 1

        meanx2 = scaled_diffs - x2_scaled.unsqueeze(-2).unsqueeze(-1)  # ... x R x M2 x M3 x K x 1
        meanx3 = scaled_diffs - x3_scaled.unsqueeze(-3).unsqueeze(-1)  # ... x R x M2 x M3 x K x 1

        Sigma = 0.5 * new_cov_inv.matmul(cov).div(self.lengthscales.pow(2).view(1, -1))  # R x K x K

        AinvdSigmaAinv = 0.25 * (new_cov_inv.unsqueeze(-2).unsqueeze(-1) *
                                 new_cov_inv.unsqueeze(-3).unsqueeze(-2)).unsqueeze(-5).unsqueeze(-5)  # R x 1 x 1 x K x K x K x K

        AinvdSigmaAinv_diag = batch_make_diag(AinvdSigmaAinv)

        AinvdSigmaAinv = AinvdSigmaAinv + AinvdSigmaAinv.transpose(-2, -1) - AinvdSigmaAinv_diag

        term0 = -new_cov_inv.unsqueeze(1).unsqueeze(1).unsqueeze(-1) * (scaled_diffs.transpose(-2, -1) -
                                                                        x2_x3_sum.div(self.lengthscales.pow(2).view(1, -1)).unsqueeze(-2)).unsqueeze(-2)

        term0_diag = batch_make_diag(term0)

        Ainvdmean = (term0 + term0.transpose(-2, -1) - term0_diag).unsqueeze(-3)

        term1 = self.dPsi2dcov(x2, mu, cov, x3).unsqueeze(-3).unsqueeze(-3) * (Sigma.unsqueeze(-3).unsqueeze(-3) +
                                                                               meanx2 * meanx3.transpose(-2, -1)).unsqueeze(-1).unsqueeze(-1)

        term2 = self.Psi2(x2, mu, cov, x3).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
            (AinvdSigmaAinv + meanx2.unsqueeze(-1).unsqueeze(-1) * Ainvdmean.transpose(-4, -3) +
             Ainvdmean * meanx3.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1))  # R x M2 x M3 x K x K x K x K

        return term1 + term2

    def Psid1d2(self, x2, mu, cov):
        """
        < d/dx d/ds k(x,s)>
        Output is R x T x M x Kgrad x K
        mu is R x 1 x K
        x2 is R x M x K
        """
        new_cov = self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # R x K x K  is (A + S)

        x2_scaled = x2.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1}

        mu_x2_sum = mu.unsqueeze(-2) + x2.matmul(cov.div(self.lengthscales.pow(2).view(-1, 1))).unsqueeze(-3)  # R x 1 x M x K

        scaled_diffs, _ = torch.solve(mu_x2_sum.unsqueeze(-1), new_cov.unsqueeze(-3).unsqueeze(-3))  # R x 1 x N2 x K x 1

        mean = scaled_diffs - x2_scaled.unsqueeze(-3).unsqueeze(-1)  # Ainv (mean - s) is R x 1 x M x K x 1

        AinvSigmaAinv, _ = torch.solve(cov.div(self.lengthscales.pow(2).view(1, -1)), new_cov)  # R x K x K

        return self.Psi1(x2, mu, cov).unsqueeze(-1).unsqueeze(-1) * ((1. / self.lengthscales.pow(2)).squeeze(1).diag().unsqueeze(0) -
                                                                     AinvSigmaAinv.unsqueeze(-3).unsqueeze(-3) - mean * mean.transpose(-2, -1))

    def dPsid1d2dmu(self, x2, mu, cov):
        """
        d/dm < d/dx d/ds k(x,s)>
        Output is R x T x M x K x K x 1 x K
        """
        new_cov = self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # R x K x K  is (A + S)

        new_cov_inv = batch_inverse_psd(new_cov)  # inv( 1/2 A + S)

        x2_scaled = x2.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1}

        mu_x2_sum = mu.unsqueeze(-2) + x2.matmul(cov.div(self.lengthscales.pow(2).view(-1, 1))).unsqueeze(-3)  # R x 1 x M x K

        scaled_diffs, _ = torch.solve(mu_x2_sum.unsqueeze(-1), new_cov.unsqueeze(-3).unsqueeze(-3))  # R x 1 x N2 x K x 1

        mean = scaled_diffs - x2_scaled.unsqueeze(-3).unsqueeze(-1)  # Ainv (mean - s) is R x 1 x M x K x 1

        AinvSigmaAinv, _ = torch.solve(cov.div(self.lengthscales.pow(2).view(1, -1)), new_cov)  # R x K x K

        dmeandm = new_cov_inv.unsqueeze(-1).unsqueeze(-1).permute(0, -1, 1, -2, 2).unsqueeze(1).unsqueeze(1)

        term1 = self.dPsi1dmu(x2, mu, cov).unsqueeze(-3).unsqueeze(-3) * \
            ((1. / self.lengthscales.pow(2)).squeeze(1).diag().unsqueeze(0) -
             AinvSigmaAinv.unsqueeze(-3).unsqueeze(-3) - mean * mean.transpose(-2, -1)).unsqueeze(-1).unsqueeze(-1)

        term2 = - self.Psi1(x2, mu, cov).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
            (mean.unsqueeze(-1).unsqueeze(-1) * dmeandm +
             dmeandm.transpose(-4, -3) * mean.unsqueeze(-1).unsqueeze(-1).transpose(-4, -3))

        return term1 + term2

    def dPsid1d2dcov(self, x2, mu, cov):
        """
        d/dS < d/dx d/ds k(x,s)>
        Output is R x T x M x K x K x K x K
        """
        new_cov = self.lengthscales.squeeze(1).diag().pow(2).unsqueeze(0) + cov  # R x K x K  is (A + S)

        new_cov_inv = batch_inverse_psd(new_cov)  # inv( 1/2 A + S)

        x2_scaled = x2.div(self.lengthscales.pow(2).view(1, -1))  # s'A^{-1}

        mu_x2_sum = mu.unsqueeze(-2) + x2.matmul(cov.div(self.lengthscales.pow(2).view(-1, 1))).unsqueeze(-3)  # R x 1 x M x K

        scaled_diffs, _ = torch.solve(mu_x2_sum.unsqueeze(-1), new_cov.unsqueeze(-3).unsqueeze(-3))  # R x 1 x N2 x K x 1

        mean = scaled_diffs - x2_scaled.unsqueeze(-3).unsqueeze(-1)  # Ainv (mean - s) is R x 1 x M x K x 1

        AinvSigmaAinv, _ = torch.solve(cov.div(self.lengthscales.pow(2).view(1, -1)), new_cov)  # R x K x K

        AinvdSigmaAinv = (new_cov_inv.unsqueeze(-2).unsqueeze(-1) *
                          new_cov_inv.unsqueeze(-3).unsqueeze(-2)).unsqueeze(-5).unsqueeze(-5)  # R x 1 x 1 x K x K x K x K

        AinvdSigmaAinv_diag = batch_make_diag(AinvdSigmaAinv)

        AinvdSigmaAinv = AinvdSigmaAinv + AinvdSigmaAinv.transpose(-2, -1) - AinvdSigmaAinv_diag

        term0 = - (new_cov_inv.unsqueeze(1).unsqueeze(1).unsqueeze(-1) * mean.squeeze(-1).unsqueeze(-2).unsqueeze(-2)).unsqueeze(-3)  # R x 1 x M x K x K x K

        term0_diag = batch_make_diag(term0)

        Ainvdmean = (term0 + term0.transpose(-2, -1) - term0_diag)

        term1 = self.dPsi1dcov(x2, mu, cov).unsqueeze(-3).unsqueeze(-3) * \
            ((1. / self.lengthscales.pow(2)).squeeze(1).diag().unsqueeze(0) -
             AinvSigmaAinv.unsqueeze(-3).unsqueeze(-3) - mean * mean.transpose(-2, -1)).unsqueeze(-1).unsqueeze(-1)

        term2 = - self.Psi1(x2, mu, cov).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
            (AinvdSigmaAinv + mean.unsqueeze(-1).unsqueeze(-1) * Ainvdmean.transpose(-3, -4) +
             Ainvdmean * mean.unsqueeze(-1).unsqueeze(-1).transpose(-3, -4))  # R x M2 x M3 x K x K x (K x K)

        return term1 + term2
