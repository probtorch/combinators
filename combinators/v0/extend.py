import torch
from torch.distributions import MultivariateNormal

def get_block(N, brange):
    if isinstance(brange, int):
        brange = (brange, brange+1)

    m = N.loc[brange[0]:brange[1]]
    S = (N.covariance_matrix[brange[0]:brange[1], brange[0]:brange[1]])
    return m, S

def get_block_complement(N, brange):
    if isinstance(brange, int):
        brange = (brange, brange+1)

    m = torch.cat((N.loc[:brange[0]], N.loc[brange[1]:]))
    Syy_1 = torch.cat((N.covariance_matrix[:brange[0], :brange[0]],
                       N.covariance_matrix[brange[1]:, :brange[0]]), dim=0)
    Syy_2 = torch.cat((N.covariance_matrix[:brange[0], brange[1]:],
                       N.covariance_matrix[brange[1]:, brange[1]:]), dim=0)
    Syy = torch.cat((Syy_1, Syy_2), dim=1)
    Sxy = torch.cat((N.covariance_matrix[brange[0]:brange[1], :brange[0]],
                     N.covariance_matrix[brange[0]:brange[1], brange[1]:]), dim=1)
    return m, Syy, Sxy

def affine_transformation(N, A, t):
    m = A @ N.loc + t
    S = A @ N.covariance_matrix @ A.t()
    return MultivariateNormal(m, S)

def marginal(N, mrange, mout=True):
    """
    Return new Gaussian marginalized over dimensions specified by mrange.
    """
    if mout:
        my, Syy, Sxy = get_block_complement(N, mrange)
    else:
        my, Syy = get_block(N, mrange)
    return MultivariateNormal(my, Syy)

def conditional(N, crange):
    """
    Return new Gaussian conditioned on dimensions specified in range crange.
    """
    mx, Sxx = get_block(N, crange)
    my, Syy, Sxy = get_block_complement(N, crange)

    Sxx_inv = Sxx.inverse()
    Syx = Sxy.t()
    S = Syy - Syx @ Sxx_inv @ Sxy
    return lambda x: MultivariateNormal(my + Syx @ Sxx_inv @ (x - mx), S)

def extend(N, F, t, B, reverse_order=False):
    return propagate(N, F, t, B, marginalize=False, reverse_order=reverse_order)

def propagate(N, F, t, B, marginalize=False, reverse_order=False):
    a = N.loc
    A = N.covariance_matrix
    b = t + F @ a
    m = torch.cat((a, b))
    FA = F @ A
    BFFA = B + F @ (FA).T
    if marginalize:
        return MultivariateNormal(loc=b, covariance_matrix=BFFA)
    if not reverse_order:
        A = N.covariance_matrix
        C1 = torch.cat((A, (FA).T), dim=1)
        C2 = torch.cat((FA, BFFA), dim=1)
        C = torch.cat((C1, C2), dim=0)
    if reverse_order:
        C1 = torch.cat((BFFA, FA), dim=1)
        C2 = torch.cat(((FA).T, A), dim=1)
        C = torch.cat((C1, C2), dim=0)
        m = torch.cat((b, a))
    return MultivariateNormal(loc=m, covariance_matrix=C)


if __name__ == '__main__':
    m = torch.arange(1., 4.)
    S = torch.tensor([[1., 0., 0.],
                      [0., 2., 0.],
                      [0., 0., 3.]])
    g = MultivariateNormal(m, S)
    # Test 1
    g1 = MultivariateNormal(m[:1], S[:1, :1])
    K12 = conditional(marginal(g, 2), 0)
    K23 = conditional(marginal(g, 0), 0)
    eval_factorized = lambda a, b, c: g1.log_prob(a) + K12(a).log_prob(b) + K23(b).log_prob(c)
    x = torch.rand(3)
    print(eval_factorized(*x), g.log_prob(x))

    # Test 2
    K12 = conditional(marginal(g, 2), 0)
    K12_3 = conditional(g, (0, 2))
    eval_factorized = lambda a, b, c: g1.log_prob(a) + K12(a).log_prob(b) + K12_3(torch.tensor([a, b])).log_prob(c)
    print(eval_factorized(*x), g.log_prob(x))

# def conditional_(mean, cov, cdims):
#     if isinstance(cdims, int):
#         cdims = (cdims, cdims+1)

#     mx = mean[cdims[0]:cdims[1]]
#     my = torch.cat((mean[:cdims[0]], mean[cdims[1]:]))
#     Sxx = (cov[cdims[0]:cdims[1], cdims[0]:cdims[1]])
#     Sxx_inv = Sxx.inverse()
#     Syy_1 = torch.cat((cov[:cdims[0], :cdims[0]],
#                        cov[cdims[1]:, :cdims[0]]), dim=0)
#     Syy_2 = torch.cat((cov[:cdims[0], cdims[1]:],
#                        cov[cdims[1]:, cdims[1]:]), dim=0)
#     Syy = torch.cat((Syy_1, Syy_2), dim=1)
#     Sxy = torch.cat((cov[cdims[0]:cdims[1], :cdims[0]],
#                      cov[cdims[0]:cdims[1], cdims[1]:]), dim=1)
#     Syx = Sxy.t()

#     S = Syy - Syx @ Sxx_inv @ Sxy
#     return lambda x: my + Syx @ Sxx_inv @ (x - mx), S

# def marginal_(mean, cov, mdims):
#     if isinstance(mdims, int):
#         mdims = (mdims, mdims+1)

#     m = mean[mdims[0]:mdims[1]]
#     S = (cov[mdims[0]:mdims[1], mdims[0]:mdims[1]])
#     return mean, cov

# def affine_transformation_(mean, cov, A, t):
#     m = A @ mean + t
#     S = A @ cov @ A.t()
#     return m, S
