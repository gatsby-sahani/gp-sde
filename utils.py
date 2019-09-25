import torch
from settings import float_type
from copy import deepcopy
import itertools


def get_points_on_grid(xdim, nperdim, xmin, xmax):
    if torch.tensor(xmin).numel() == 1 and torch.tensor(xmax).numel() == 1 and torch.tensor(nperdim).numel() == 1:
        gridLinSpace = torch.linspace(xmin, xmax, int(nperdim))
        all_gridLinSpaces = [gridLinSpace] * xdim
    else:
        all_gridLinSpaces = [torch.linspace(xmin[i], xmax[i], int(nperdim[i])) for i in range(nperdim.numel())]

    xgrid = torch.tensor(list(itertools.product(*all_gridLinSpaces)))
    return xgrid


def create_test_grid(n_test_grid=32, ndims=2, device='cpu', a=0.0, b=1.0):
    if torch.tensor(a).numel() == 1 and torch.tensor(n_test_grid).numel() == 1:
        gridLinSpace = torch.linspace(a, b, int(n_test_grid))
        all_gridLinSpaces = [gridLinSpace] * ndims
    else:
        all_gridLinSpaces = [torch.linspace(a[i], b[i], int(n_test_grid[i])) for i in range(n_test_grid.numel())]

    test_x = torch.tensor(list(itertools.product(*all_gridLinSpaces))).to(device)
    return test_x


def create_copy_with_grad(*objs):
    """ Create a copy of each input and set requires_grad on
    """
    out = list()
    for obj in objs:
        obj_copy = deepcopy(obj.data)
        obj_copy.requires_grad = True
        out.append(obj_copy)

    return out


def stack_along_dim(tensor, dim_unbind, dim_stack):
    """
    take elements of tensor along dim_stack and stack them along dim_unbind
    this is equivalent to calling
    torch.cat(torch.unbind(tensor, dim=dim_unbind), dim=dim_stack)
    IMPORTANT: dimension input needs to be from the back of tensor, i.e. -1,-2 etc.
    such that this works with different batch sizes. Otherwise the code breaks.
    """
    # permute tensor as [..., dim_unbind, dim_stack]

    permuted_tensor = tensor.transpose(dim_stack, -1).transpose(dim_unbind, -2)
    perm_size_tuple = tuple(permuted_tensor.size())
    return permuted_tensor.reshape(*perm_size_tuple[:-2], -1).transpose(dim_unbind + 1, -1)


def get_diag_of_tensor(A, dim1, dim2):
    A = A.transpose(0, dim1).transpose(1, dim2)
    A = torch.stack([A[i][i] for i in range(min(A.size(0), A.size(1)))], dim=0)
    return A.transpose(1, max(dim2 - 1, 0)).transpose(0, max(dim1 - 1, 0))


def get_all_grads(target, *args, within_batch=True):
    # Make sure all args require grad
    # target.requires_grad = True
    # args = list(map(lambda x: x.requires_grad_(True), args))

    target_1d = target.contiguous().view(-1)
    basis_vecs = torch.eye(target_1d.size(0))

    # Get the gradients with respect to each element of target_1d using a loop

    a = [list(torch.autograd.grad(target_1d, args, tensor_element, retain_graph=True))

         for tensor_element in torch.unbind(basis_vecs, dim=1)]

    a = list(map(list, zip(*a)))

    a = [torch.stack(all_grads, dim=0) for all_grads in a]

    a = [all_grads.view(*(target.shape + all_grads.shape[1:])) for all_grads in a]

    if within_batch:
        a = [get_diag_of_tensor(all_grads, 0, target.ndimension()) for all_grads in a]

    return a


def expand_matrix_as(A, B):
    """
    Expand matrix A (N x M) to work with bmm operation with B (R x U x N)

    outputs A as R x N x M
    """

    return A.unsqueeze(0).expand((B.size(0), -1, -1))


def unsqueeze_as(A, B, dim=0):
    """
    Expand tensor A to match size of B by unsqueezing batch dimensions
    """
    size_diff = len(list(B.size())) - len(list(A.size()))
    if dim == 0:
        A_unsqueezed = A[(None,) * size_diff]
    elif dim == -1:
        A_unsqueezed = A[(...,) + (None,) * size_diff]
    else:
        raise('can only unsqueeze along first (0) or last (-1) dim')
    return A_unsqueezed


def neg_variational_free_energy(ell, kld, prior_trans, prior_map):
    nvfe = - ell + kld - prior_trans - prior_map
    return nvfe


def apply_along_axis(func, M, dim):
    tList = [func(m) for m in torch.unbind(M, dim)]
    res = torch.stack(tList, dim).to(device=M.device)
    return res


def apply_along_batch_axis(func, M):
    """
    N x ... x M x L x M tensor input, applies func to L x M matrices in tensor
    Function must not change tensor size, i.e. det(M) -> scalar of size 1x1
    """
    Msize = tuple(M.size())
    tList = [func(m) for m in torch.unbind(M.view(-1, Msize[-2], Msize[-1]), dim=0)]
    res = torch.stack(tList, dim=0).to(device=M.device)  # 3 D tensor
    return res.view(*Msize[:-2], res.size(-2), res.size(-1))


def det_keepdim(A):
    return torch.det(A).view(1, 1)


def logdet(M):
    # L = torch.potrf(M, upper=False)
    L = torch.cholesky(M)
    return 2 * torch.diag(L).log().sum().unsqueeze(-1).unsqueeze(-1)  # no collapse


def batch_colesky(A):
    """
    batch implementation of cholesky factorisation
    """
    L = torch.zeros_like(A)

    for i in range(A.shape[-1]):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s = s + L[..., i, k].clone() * L[..., j, k].clone()

            L[..., i, j] = torch.sqrt(A[..., i, i] - s) if (i == j) else \
                (1.0 / L[..., j, j].clone() * (A[..., i, j] - s))
    return L


def batch_inverse_tril(L):
    # Batched inverse of lower triangular matrices
    n = L.shape[-1]
    invL = torch.zeros_like(L)
    for j in range(0, n):
        invL[..., j, j] = 1.0 / L[..., j, j]
        for i in range(j + 1, n):
            S = 0.0
            for k in range(i + 1):
                S = S - L[..., i, k] * invL[..., k, j].clone()
            invL[..., i, j] = S / L[..., i, i]

    return invL


def batch_inverse_psd(A):
    """
    matrix inverse of postitive definite Matrices using cholesky decomposition
    """
    sz = A.shape
    # check for square matrix
    if sz[-1] != sz[-2]:
        raise('Error: can only take determinant of batches of square matrices')

    # Check if we get a batch of 1x1 matrices, then just return the reciprocal
    if sz[-2] == 1:
        return 1. / A

    # Check if we get a batch of 2x2 matrices, then just return the inverse computed by hand

    if sz[-2] == 2:
        return batch_inverse_2d(A)

    # otherwise get inverse via cholesky
    # L = torch.cholesky(A)
    L = batch_colesky(A)
    Linv = batch_inverse_tril(L)
    Ainv = Linv.transpose(-2, -1).matmul(Linv)
    return Ainv


def batch_det_psd(A, keepdim=True):
    """
    batch determinant of a positive definite matrix
    """
    sz = A.shape
    if sz[-1] != sz[-2]:
        raise('Error: can only take determinant of batches of square matrices')
    # Check if we get a batch of 1x1 matrices, then just return them
    if sz[-2] == 1:
        return A
    # Check if we get a batch of 2x2 matrices, then just compute the determinant by hand
    if sz[-2] == 2:
        return batch_det_2d(A, keepdim)
    # otherwise get inverse via cholesky
    L = batch_colesky(A)
    Asize = tuple(A.size())
    Adet = torch.prod(batch_diag(L)**2, dim=-1, keepdim=keepdim).view(*Asize[:-2], 1, 1)
    return Adet


def batch_inverse(A):
    # does linear solve to get batch inverse of a tensor
    # eyemat = A.new_ones(A.size(-1)).diag().expand_as(A)
    sz = A.shape

    # check for square matrix
    if sz[-1] != sz[-2]:
        raise('Error: can only take inverse of batches of square matrices')

    # Check if we get a batch of 1x1 matrices, then just return the reciprocal
    if sz[-2] == 1:
        return 1. / A

    # Check if we get a batch of 2x2 matrices, then just return the reciprocal

    if sz[-2] == 2:
        return batch_inverse_2d(A)

    A_inv = torch.inverse(A)
    return A_inv


def batch_inverse_2d(A):
    detA = batch_det_2d(A, keepdim=True)
    matA = -A[:]
    matA[..., 0, 0] = A[..., 1, 1]
    matA[..., 1, 1] = A[..., 0, 0]
    return matA.div(detA)


def batch_det(A, keepdim=True):
    # Input A is B x N x N tensor, this returns determinants for each NxN matrix based on batch LU factorisation

    # Testing:
    # B = 2000
    # N = 20
    # A = torch.randn(B, N, N)
    # A = (A + A.transpose(-1,-2))/2 # Symmetric
    # torch.allclose(apply_along_axis(torch.det, A, dim=0), batch_det(A))

    # Based on what https://pytorch.org/docs/stable/_modules/torch/functional.html - btriunpack does:
    # The diagonals of each A_LU are the diagonals of U (so det(U) = prod(diag(A_LU[n])))
    # The diagonals of L are not given, but are fixed to all 1s (so det(L) = 1)
    # The pivots determine the permutation matrix that left-multiplies A (i.e. switches rows)
    # Therefore to get the final determinant, we need to get det(L)*det(U) * (-1^[how many times P switches rows])

    sz = A.shape

    if sz[-1] != sz[-2]:
        raise('Error: can only take determinant of batches of square matrices')

    # Check if we get a batch of 1x1 matrices, then just return them
    if sz[-2] == 1:
        return A

    # Check if we get a batch of 2x2 matrices, then just compute the determinant by hand
    if sz[-2] == 2:
        return batch_det_2d(A, keepdim)

    # if multibatchaxes:
    A = A.view(-1, A.size(-2), A.size(-1))

    A_LU, pivots = torch.btrifact(A)

    # detL = 1
    detU = batch_diag(A_LU).prod(1)
    detP = (-1. * A.new_ones(A.size(0))).pow(
        ((pivots - (torch.arange(A.size(1), dtype=torch.int, device=A.device) + 1).expand(A.size(0), A.size(1))) != 0).sum(1).float())

    if keepdim:
        return (detU * detP).view(*sz[:-2], 1, 1)
    else:
        return (detU * detP).view(*sz[:-2])


def batch_det_2d(A, keepdim):
    # computes determinant of a batch of 2 x 2 matrices
    Adet = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    if keepdim is True:
        return Adet.unsqueeze(-1).unsqueeze(-1)
    return Adet


def batch_diag(A):
    # extracts diagonal of square matrix along batch dimension
    dim1 = A.size(-1)
    dim2 = A.size(-2)

    assert(dim1 == dim2)

    idx = torch.arange(0, dim1, out=torch.LongTensor())

    return A[..., idx, idx]


def batch_make_diag(A):
    # extracts diagonal of square matrix along batch dimension
    dim1 = A.size(-1)
    dim2 = A.size(-2)
    # assert square slices
    assert(dim1 == dim2)
    Adiag = torch.zeros_like(A)
    idx = torch.arange(0, dim1, out=torch.LongTensor())
    Adiag[..., idx, idx] = A[..., idx, idx][:]
    return Adiag


def batch_vec_to_diag(A):
    # A is ... x K x M
    # returns ... x K x M x M batch of diagonal matrices
    sz = tuple(A.size())
    Adiag = torch.zeros(*sz, sz[-1]).type(float_type)
    idx = torch.arange(0, sz[-1], out=torch.LongTensor())
    Adiag[..., idx, idx] = A[:]
    return Adiag


def linInterp(t, A, t_grid, asnumpy=False):
    # function to linearly interpolate value between two points in tensor
    if t <= t_grid.min():  # use only first grid point
        Astart = A[0, ]
        Astop = A[0, ]
        weight = 1.
    elif t >= t_grid.max():  # use only last grid point
        Astart = A[-1, ]
        Astop = A[-1, ]
        weight = 1.
    else:
        tstart = t_grid[t_grid <= t].max()
        tstop = t_grid[t_grid >= t].min()
        idxstart = (tstart == t_grid).nonzero()
        idxstop = (tstop == t_grid).nonzero()

        Astart = A[idxstart, ]
        Astop = A[idxstop, ]

        if idxstart == idxstop:
            weight = 1.
        else:
            weight = ((t - tstart) / (tstop - tstart))

        if asnumpy:
            weight = weight.numpy()

    return Astart + weight * (Astop - Astart)


def bin_spikeTrain(Y, Tmax, dtstep):
    nbins = int(Tmax / dtstep)
    nNeur = len(Y)
    Ybin = torch.zeros(nbins, nNeur).type(float_type)
    for ii in range(nNeur):
        Ybin[:, ii] = torch.histc(torch.tensor(Y[ii]), bins=nbins, min=0, max=Tmax)
    return Ybin


def bin_sparseData(Y, tObs, Tmax, dtstep):
    # Y is a T x D tensor
    # tobs is list of sample times
    nbins = int(Tmax / dtstep)
    nDims = Y.size(-1)
    obsMask = torch.histc(tObs, bins=nbins, min=0, max=Tmax).type(torch.ByteTensor)
    Ybin = torch.zeros(nbins, nDims).type(float_type)
    Ybin[obsMask.unsqueeze(-1).expand(Ybin.size())] = Y.view(-1)
    return Ybin


def simple_grad_check(fun, grad, x0):
    tol = 1e-6
    rr = torch.randn(x0.size()) * tol
    finite_diffs = fun(x0 + rr / 2) - fun(x0 - rr / 2)
    print('finite differences:{}  analytic gradient:{}'.format(finite_diffs.item(), (rr * grad(x0)).sum().item()))


def get_grad(optimizer, model, inputs):
    """
    Computes objective and gradient of pytorch model
        optimizer (Optimizer): the PBQN optimizer
        model: forward pass computes loss
        inputs: tuple of inputs model expects
    Outputs:
        grad (tensor): stochastic gradient over sample Sk
        obj (tensor): stochastic function value over sample Sk
    """
    optimizer.zero_grad()

    # forward pass
    ell, kld, prior_trans, prior_map = model(*inputs)  # unpacks tuple of inputs into arguments

    # define loss and perform backward pass
    loss = neg_variational_free_energy(ell, kld, prior_trans, prior_map)
    loss.backward()
    # gather flat gradient
    grad = optimizer._gather_flat_grad()

    return grad, loss


def train_model(model, inputs, maxiter=100):

    # model is model object containing parameters we want to optimize
    # inputs is a tuple of all the inputs that will be passed into model forward pass

    optimizer = torch.optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()), max_iter=maxiter)
    # Forward pass: Compute predicted mean and varince by passing input to the model
    ell, kld, prior_trans, prior_map = model(*inputs)
    loss = neg_variational_free_energy(ell, kld, prior_trans, prior_map)

    def closure():
        optimizer.zero_grad()
        # Forward pass: Compute predicted mean and varince by passing input to the model
        ell, kld, prior_trans, prior_map = model(*inputs)  # unpacks tuple of inputs into arguments
        # Compute loss
        loss = neg_variational_free_energy(ell, kld, prior_trans, prior_map)
        loss.backward()

        return loss

    loss = optimizer.step(closure)

    # return final loss after learning update
    ell, kld, prior_trans, prior_map = model(*inputs)
    return ell.item(), kld.item(), prior_trans.item(), prior_map.item()
