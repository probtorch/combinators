import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from experiments.apgs_gmm.objectives import posterior_eta
from combinators.stochastic import Trace


import torch.distributions as D
import torch.distributions as Dist
from collections import namedtuple
from combinators import Program, Kernel
from combinators.densities import OneHotCategorical, OneHotCategorical, NormalGamma
from combinators.stochastic import Trace, RandomVariable, Provenance
from combinators.trace.utils import maybe_sample


import torch
from combinators.densities import RingGMM

ix = namedtuple('ix', ['sweep', 'rev', 'block'])

getmap = lambda name: (lambda ix: (f'{name}{ix.sweep if ix.rev else ix.sweep-1}', f'{name}{ix.sweep-1 if ix.rev else ix.sweep}'))
pmap = getmap('precisions')
mmap = getmap('means')
smap = getmap('states')

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

    def model(self, trace, x, prior_ng, ix=None):
        assert ix is not None
        if ix == '':
            (prior_alpha, prior_beta, prior_mu, prior_nu) = prior_ng
            q_alpha, q_beta, q_mu, q_nu = posterior_eta(self.rws_eta_ob(x) , self.rws_eta_gamma(x), prior_alpha, prior_beta, prior_mu, prior_nu)
            tau = Gamma(q_alpha, q_beta).sample()
            trace.gamma(q_alpha, q_beta, value=tau, name='precisions0')
            mu = Normal(q_mu, 1. / (q_nu * trace['precisions0'].value).sqrt()).sample()
            trace.normal(q_mu, 1. / (q_nu * trace['precisions0'].value).sqrt(), # std = 1 / sqrt(nu * tau)
                    value=mu, name='means0')
        else:
            (prior_alpha, prior_beta, prior_mu, prior_nu) = prior_ng
            q_alpha, q_beta, q_mu, q_nu = posterior_eta(self.rws_eta_ob(x) , self.rws_eta_gamma(x), prior_alpha, prior_beta, prior_mu, prior_nu)
            tau = Gamma(q_alpha, q_beta).sample()
            trace.gamma(q_alpha, q_beta, value=tau, name='precisions1')
            mu = Normal(q_mu, 1. / (q_nu * trace['precisions1'].value).sqrt()).sample()
            trace.normal(q_mu, 1. / (q_nu * trace['precisions1'].value).sqrt(), # std = 1 / sqrt(nu * tau)
                    value=mu, name='means1')
        return None


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

    def apply_kernel(self, q_eta_z_new, q_eta_z, q_output, x, prior_ng, ix=None):
        # Need to confirm that: q(η_2 | z_1 x) p_{prp1}(x η_1 z_1)
        assert ix is not None
        (prior_alpha, prior_beta, prior_mu, prior_nu) = prior_ng
        if ix is None:
            breakpoint();
            raise ValueError

        if ix == '':
            try:
                z = q_eta_z['states0'].value
            except:
                raise ValueError
            # in "hao mode"
            ob_z = torch.cat((x, z), -1) # concatenate observations and cluster asssignemnts
            q_alpha, q_beta, q_mu, q_nu = posterior_eta(self.apg_eta_ob(ob_z) , self.apg_eta_gamma(ob_z), prior_alpha, prior_beta, prior_mu, prior_nu)
            _ = q_eta_z_new.one_hot_categorical(probs=q_eta_z['states0'].dist.probs, value=q_eta_z['states0'].value, name='states0')
            if 'means0' in q_eta_z.keys():
                q_eta_z_new.gamma(q_alpha, q_beta, value=q_eta_z['precisions0'].value, name='precisions0')
                q_eta_z_new.normal(q_mu, 1. / (q_nu * q_eta_z['precisions0'].value).sqrt(), value=q_eta_z['means0'].value, name='means0')
            else:
                tau = Gamma(q_alpha, q_beta).sample()
                q_eta_z_new.gamma(q_alpha, q_beta, value=tau, name='precisions0')
                mu = Normal(q_mu, 1. / (q_nu * tau).sqrt()).sample()
                q_eta_z_new.normal(q_mu, 1. / (q_nu * tau).sqrt(), value=mu, name='means0')
        else:
            zfr, _ = smap(ix)

            # leave z's unmoved
            z = q_eta_z[zfr].value
            ob_z = torch.cat((x, z), -1) # concatenate observations and cluster asssignemnts
            q_alpha, q_beta, q_mu, q_nu = posterior_eta(self.apg_eta_ob(ob_z) , self.apg_eta_gamma(ob_z), prior_alpha, prior_beta, prior_mu, prior_nu)
            _ = q_eta_z_new.one_hot_categorical(probs=q_eta_z[zfr].dist.probs, value=q_eta_z[zfr].value, name=zfr)

            # move etas forward or backwards
            pfr, pto = pmap(ix)
            gamma = Gamma(q_alpha, q_beta)
            tau = q_eta_z[pfr].value if pfr in q_eta_z else gamma.sample()
            provenance = Provenance.OBSERVED if pfr in q_eta_z else Provenance.SAMPLED
            # log_prob = q_eta_z[pto].log_prob if pfr in q_eta_z else None
            q_eta_z_new.append(RandomVariable(dist=gamma, value=tau, provenance=provenance), name=pto) #, log_prob=log_prob))

            mfr, mto = mmap(ix)
            normal = Normal(q_mu, 1. / (q_nu * tau).sqrt())
            mu = q_eta_z[mfr].value if mfr in q_eta_z else normal.sample()
            provenance = Provenance.OBSERVED if mfr in q_eta_z else Provenance.SAMPLED
            # log_prob = q_eta_z[mto].log_prob if mfr in q_eta_z else None
            q_eta_z_new.append(RandomVariable(dist=normal, value=mu, provenance=provenance), name=mto) #, log_prob=log_prob))

        return None


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

    def apply_kernel(self, trace, q_eta_z, q_output, x, ix=None):
        # (x, prior_ng) = q_output
        S, B, N, D = x.shape
        assert ix is not None
        if ix == '':
            try:
                tau = q_eta_z['precisions0'].value
                mu = q_eta_z['means0'].value
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
            q_eta_z_new.gamma(q_eta_z['precisions0'].dist.concentration,
                              q_eta_z['precisions0'].dist.rate,
                              value=q_eta_z['precisions0'].value,
                              name='precisions0')
            q_eta_z_new.normal(q_eta_z['means0'].dist.loc,
                               q_eta_z['means0'].dist.scale,
                               value=q_eta_z['means0'].value,
                               name='means0')

            if 'states0' in q_eta_z.keys():
                _ = q_eta_z_new.one_hot_categorical(probs=q_probs, value=q_eta_z['states0'].value, name='states1')
            else:
                _ = q_eta_z_new.one_hot_categorical(probs=q_probs, name='states0')
            return None
        else:
            pfr, _ = pmap(ix)
            mfr, _ = mmap(ix)
            if ix.block == 'is':
                pfr, mfr = 'precisions1', 'means1'
            assert pfr in q_eta_z

            tau = q_eta_z[pfr].value
            mu = q_eta_z[mfr].value

            q_eta_z_new = trace
            q_eta_z_new.gamma(q_eta_z[pfr].dist.concentration, q_eta_z[pfr].dist.rate, value=tau, name=pfr)
            q_eta_z_new.normal(q_eta_z[mfr].dist.loc, q_eta_z[mfr].dist.scale, value=mu, name=mfr)

            # ===================================
            K = mu.shape[-2]
            mu_expand = mu.unsqueeze(2).repeat(1, 1, N, 1, 1)
            tau_expand = tau.unsqueeze(2).repeat(1, 1, N, 1, 1)
            x_expand = x.unsqueeze(-2).repeat(1, 1, 1, K, 1)
            var = torch.cat((x_expand, mu_expand, tau_expand), -1)
            assert var.shape == (S, B, N, K, 3*D)

            gamma_list = self.pi_log_prob(var).squeeze(-1)
            q_probs = F.softmax(gamma_list, -1)
            _, zto = smap(ix)

            ohcat = Dist.OneHotCategorical(probs=q_probs)
            value = q_eta_z[zto].value if zto in q_eta_z else ohcat.sample()
            log_prob = q_eta_z[zto].log_prob if zto in q_eta_z else None
            q_eta_z_new.append(RandomVariable(dist=ohcat, value=value, log_prob=log_prob), name=zto)

            breakpoint();
            return None


class GenerativeOriginal(Program):
    """
    The generative model of GMM
    """
    def __init__(self, K, D, CUDA, device):
        super().__init__(with_joint={'precisions0', 'means0', 'states0', 'lls0'})
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
        tau = q['precisions0'].value
        mu = q['means0'].value
        z = q['states0'].value

        p = Trace()
        p.gamma(self.prior_alpha, self.prior_beta, value=tau, name='precisions0')
        p.normal(self.prior_mu, 1. / (self.prior_nu * tau).sqrt(), value=mu, name='means0')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='states0')

        labels_flat = z.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])
        mu_expand = torch.gather(mu, 2, labels_flat)
        sigma_expand = torch.gather(1. / tau.sqrt(), 2, labels_flat)
        p.normal(mu_expand, sigma_expand, value=x, name='lls0')
        return p

    def model(self, trace, x, reparameterized=False, ix=None):
        """ evaluate the log joint i.e. log p (x, z, tau, mu) """
        assert ix is not None
        ptar, mtar, star = 'precisions0', 'means0', 'states0'
        llstar = "lls0"
        if ix.block == 'is':
            p, m, s = 'precisions1', 'means1', 'states1'
        else:
            breakpoint();

        assert p in trace and m in trace and s in trace
        provenance=Provenance.OBSERVED

        gamma = D.Gamma(self.prior_alpha, self.prior_beta)
        tau = trace[p].value
        trace.append(RandomVariable(dist=gamma, value=tau, provenance=provenance), name=ptar)

        normal = D.Normal(self.prior_mu, 1. / (self.prior_nu * tau).sqrt())
        mu = trace[m].value
        trace.append(RandomVariable(dist=normal, value=mu, provenance=provenance), name=mtar)
        oh_cat = D.OneHotCategorical(probs=self.prior_pi)
        z = trace[s].value
        trace.append(RandomVariable(dist=oh_cat, value=z, provenance=provenance), name=star)

        labels_flat = z.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])
        mu_expand = torch.gather(mu, 2, labels_flat)
        sigma_expand = torch.gather(1. / tau.sqrt(), 2, labels_flat)
        trace.normal(mu_expand, sigma_expand, value=x, name=llstar)

        return None

