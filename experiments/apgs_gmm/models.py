import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from experiments.apgs_gmm.objectives import posterior_eta
from combinators.stochastic import Trace


import torch.distributions as D
from collections import namedtuple
from combinators import Program, Kernel
from combinators.densities import OneHotCategorical, OneHotCategorical, NormalGamma
from combinators.stochastic import Trace, RandomVariable, Provenance
from combinators.trace.utils import maybe_sample


import torch
from combinators.densities import RingGMM

PRECISIONS='precision'
MEANS='means'
STATES='states'

def mk_target(normal_gamma_priors):
    return Generative(num_clusters=3, normal_gamma_priors=normal_gamma_priors)

def mk_nvi_target():
    return RingGMM(loc_scale=6, scale=1, count=3, name="target")

class Enc_rws_eta(Program):
    """
    One-shot (i.e.RWS) encoder of {mean, covariance} i.e. \eta
    input i.e. a batch of GMM of size (S, B, N, 2)
    """
    def __init__(self, K, D):
        super().__init__()

        self.rws_eta_gamma = nn.Sequential(
            nn.Linear(D, K),
            nn.Softmax(-1))

        self.rws_eta_ob = nn.Sequential(
            nn.Linear(D, D))

    def model(self, trace, x, prior_ng):
        (prior_alpha, prior_beta, prior_mu, prior_nu) = prior_ng
        q_alpha, q_beta, q_mu, q_nu = posterior_eta(self.rws_eta_ob(x) , self.rws_eta_gamma(x), prior_alpha, prior_beta, prior_mu, prior_nu)
        tau = Gamma(q_alpha, q_beta).sample()
        trace.gamma(q_alpha,
                q_beta,
                value=tau,
                name='precisions')
        mu = Normal(q_mu, 1. / (q_nu * trace['precisions'].value).sqrt()).sample()
        trace.normal(q_mu,
                1. / (q_nu * trace['precisions'].value).sqrt(), # std = 1 / sqrt(nu * tau)
                value=mu,
                name='means')
        return (x, prior_ng)


class Enc_apg_eta(Kernel):
    """
    Conditional proposal of {mean, covariance} i.e. \eta
    """
    def __init__(self, K, D):
        super().__init__()
        self.apg_eta_gamma = nn.Sequential(
            nn.Linear(K+D, K),
            nn.Softmax(-1))
        self.apg_eta_ob = nn.Sequential(
            nn.Linear(K+D, D))

    def apply_kernel(self, q_eta_z_new, q_eta_z, q_output):
        x, prior_ng = q_output
        (prior_alpha, prior_beta, prior_mu, prior_nu) = prior_ng
        try:
            z = q_eta_z['states'].value
        except:
            raise ValueError

        ob_z = torch.cat((x, z), -1) # concatenate observations and cluster asssignemnts
        q_alpha, q_beta, q_mu, q_nu = posterior_eta(self.apg_eta_ob(ob_z) , self.apg_eta_gamma(ob_z), prior_alpha, prior_beta, prior_mu, prior_nu)

        _ = q_eta_z_new.one_hot_categorical(probs=q_eta_z['states'].dist.probs, value=q_eta_z['states'].value, name='states')

        gamma = Gamma(q_alpha, q_beta).sample()
        tau, provenance = maybe_sample(q_eta_z, None)(gamma, 'precisions')
        q_eta_z_new.append(RandomVariable(dist=gamma, value=tau, provenance=provenance), name='precisions')

        normal = Normal(q_mu, 1. / (q_nu * tau).sqrt())
        mu, provenance = maybe_sample(q_eta_z, None)(normal, 'means')
        q_eta_z_new.append(RandomVariable(dist=normal, value=mu, provenance=provenance), name='means')

        return x # for Enc_apg_z.apply_kernel(..., q_output)



class Enc_apg_z(Kernel):
    """
    Conditional proposal of cluster assignments z
    """
    def __init__(self, K, D, num_hidden):
        super().__init__()
        self.pi_log_prob = nn.Sequential(
            nn.Linear(3*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

    def apply_kernel(self, trace, q_eta_z, q_output):
        (x, prior_ng) = q_output
        S, B, N, D = x.shape
        try:
            tau = q_eta_z['precisions'].value
            mu = q_eta_z['means'].value
        except:
            raise ValueError

        K = mu.shape[-2]
        mu_expand = mu.unsqueeze(2).repeat(1, 1, N, 1, 1)
        tau_expand = tau.unsqueeze(2).repeat(1, 1, N, 1, 1)
        x_expand = x.unsqueeze(-2).repeat(1, 1, 1, K, 1)
        var = torch.cat((x_expand, mu_expand, tau_expand), -1)
        assert var.shape == (S, B, N, K, 3*D)

        gamma_list = self.pi_log_prob(var).squeeze(-1)
        q_probs = F.softmax(gamma_list, -1)

        q_eta_z_new = trace
        q_eta_z_new.gamma(q_eta_z['precisions'].dist.concentration,
                          q_eta_z['precisions'].dist.rate,
                          value=q_eta_z['precisions'].value,
                          name='precisions')
        q_eta_z_new.normal(q_eta_z['means'].dist.loc,
                           q_eta_z['means'].dist.scale,
                           value=q_eta_z['means'].value,
                           name='means')

        if 'states' in q_eta_z.keys():
            _ = q_eta_z_new.one_hot_categorical(probs=q_probs, value=q_eta_z['states'].value, name=STATES)
        else:
            _ = q_eta_z_new.one_hot_categorical(probs=q_probs, name=STATES)
        return None


class GenerativeOriginal(Program):
    """
    The generative model of GMM
    """
    def __init__(self, K, D, CUDA, device):
        super().__init__(with_joint=True)
        self.K = K
        self.prior_mu = torch.zeros((K, D))
        self.prior_nu = torch.ones((K, D)) * 0.1
        self.prior_alpha = torch.ones((K, D)) * 2
        self.prior_beta = torch.ones((K, D)) * 2
        self.prior_pi = torch.ones(K) * (1./ K)

        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu = self.prior_mu.cuda()
                self.prior_nu = self.prior_nu.cuda()
                self.prior_alpha = self.prior_alpha.cuda()
                self.prior_beta = self.prior_beta.cuda()
                self.prior_pi = self.prior_pi.cuda()

        self.prior_ng = (self.prior_alpha, self.prior_beta, self.prior_mu, self.prior_nu) ## this tuple is needed as parameter in enc_eta as enc_rws

    def x_forward(self, q, x):
        """
        evaluate the log joint i.e. log p (x, z, tau, mu)
        """
        p = Trace()
        tau = q['precisions'].value
        mu = q['means'].value
        z = q['states'].value

        p.gamma(self.prior_alpha, self.prior_beta, value=tau, name='precisions')
        p.normal(self.prior_mu, 1. / (self.prior_nu * tau).sqrt(), value=mu, name='means')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='states')

        labels_flat = z.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])
        mu_expand = torch.gather(mu, 2, labels_flat)
        sigma_expand = torch.gather(1. / tau.sqrt(), 2, labels_flat)
        p.normal(mu_expand, sigma_expand, value=x, name='lls')
        return p

    def model(self, trace, x, reparameterized=False):
        """
        evaluate the log joint i.e. log p (x, z, tau, mu)
        """
        assert 'precisions' in trace and 'means' in trace and 'states' in trace
        provenance=Provenance.OBSERVED

        gamma = D.Gamma(self.prior_alpha, self.prior_beta)
        tau = trace['precisions'].value
        trace.append(RandomVariable(dist=gamma, value=tau, provenance=provenance), name='precisions')

        normal = D.Normal(self.prior_mu, 1. / (self.prior_nu * tau).sqrt())
        mu = trace['means'].value
        trace.append(RandomVariable(dist=normal, value=mu, provenance=provenance), name='means')

        oh_cat = D.OneHotCategorical(probs=self.prior_pi)
        z = trace['states'].value
        trace.append(RandomVariable(dist=oh_cat, value=z, provenance=provenance), name='states')

        labels_flat = z.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])
        mu_expand = torch.gather(mu, 2, labels_flat)
        sigma_expand = torch.gather(1. / tau.sqrt(), 2, labels_flat)
        trace.normal(mu_expand, sigma_expand, value=x, name='lls')

        return None


class Generative(Program):
    """ The generative model of GMM """
    def __init__(self, num_clusters, normal_gamma_priors):
        super().__init__()
        self.K = num_clusters
        self.normal_gamma = NormalGamma(**normal_gamma_priors)
        self.assignments = OneHotCategorical(name='assignments', probs=torch.ones(self.K) * (1./ self.K))
        self.Output = namedtuple('GMMOutput', ['observations', 'precisions', 'means', 'assignments'])
        self.prior_ng = (
            self.normal_gamma.mu,
            self.normal_gamma.nu,
            self.normal_gamma.alpha,
            self.normal_gamma.beta,
        )
    def model(self, p, q, x):
        """
        evaluate the log joint i.e. log p (x, z, tau, mu)
        """
        tau = q['precisions'].value
        mu = q['means'].value
        z = q['states'].value

        print('tau', tau.shape)
        print('mu', mu.shape)
        print('z', z.shape)

        p.gamma(self.prior_alpha, self.prior_beta, value=tau, name='precisions')
        p.normal(self.prior_mu, 1. / (self.prior_nu * tau).sqrt(), value=mu, name='means')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='states')

        labels_flat = z.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])
        mu_expand = torch.gather(mu, 2, labels_flat)
        sigma_expand = torch.gather(1. / tau.sqrt(), 2, labels_flat)
        p.normal(mu_expand, sigma_expand, value=x, name='lls')
        return p

    def model(self, trace, x, sample_shape=None):
        broadcast = lambda o: (o.trace, o.output)
        tr_ng, (tau, mu) = broadcast(self.normal_gamma(sample_shape=sample_shape))
        tr_z, z = broadcast(self.assignments(sample_shape=sample_shape))
        trace.update(tr_ng)
        trace.update(tr_z)

        if True:
            labels_flat = z.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])
            mu_expand = torch.gather(mu, 2, labels_flat)
            sigma_expand = torch.gather(1. / tau.sqrt(), 2, labels_flat)
            normal = Normal(mu_expand, sigma_expand)
        else:
            labels = z.argmax(-1)
            sigma = 1. / torch.sqrt(tau)

            if len(labels.shape) == 1:
                normal = Normal(means[labels], sigma[labels])
            else:
                assert len(labels.shape) == 2, "assume only 2 dims for now"
                normal = Normal(means[labels[:], labels[:]], sigma[labels[:], labels[:]])

        LLS = 'lls'
        if x is None:
            x, provenance = maybe_sample(trace, None)(normal, LLS)
            trace.append(RandomVariable(dist=normal, value=x, provenance=provenance), name=LLS)
        else:
            trace.append(RandomVariable(dist=normal, value=x, provenance=Provenance.OBSERVED), name=LLS)

        return self.Output(x, tau, mu, z)

    def log_likelihood(self, xs, z, tau, mu):
        """
        aggregate = False : return S * B * N
        aggregate = True : return S * B * K
        """
        sigma = 1. / tau.sqrt()
        labels = z.argmax(-1)
        labels_flat = labels.unsqueeze(-1).repeat(1, 1, 1, xs.shape[-1])
        mu_expand = torch.gather(mu, 2, labels_flat)
        sigma_expand = torch.gather(sigma, 2, labels_flat)
        ll = Normal(mu_expand, sigma_expand).log_prob(xs).sum(-1) # S * B * N
        # if aggregate:
        #     ll = ll.sum(-1) # S * B
        return ll


