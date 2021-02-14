import os
import torch
import numpy as np
# import matplotlib.pyplot as plt
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat

from experiments.apgs_gmm.evaluation import plot_cov_ellipse

class Sim_GMM():
    """
    simulate instances of GMM
    N : number of points
    K : number of clusters
    D : data point dim
    alpha, beta, mu, nu are parameters of normal-gamma
    """
    def __init__(self, N, K, D, alpha, beta, mu, nu):
        super().__init__()

        self.N = N
        self.K = K
        self.D = D
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu = nu

    def sim_one_gmm(self):
        precision = Gamma(torch.ones((self.K, self.D)) * self.alpha, torch.ones((self.K, self.D)) * self.beta).sample()
        sigma_of_mean = 1. / (precision * self.nu).sqrt()
        sigma = 1. / torch.sqrt(precision)
        mean = Normal(torch.ones((self.K, self.D)) * self.mu, sigma_of_mean).sample()
        assignment = cat(torch.ones(self.K) * (1. / self.K)).sample((self.N,))
        labels = assignment.argmax(-1)

        ob = Normal(mean[labels], sigma[labels]).sample()
        return ob.data.numpy(), precision.data.numpy(), mean.data.numpy(), assignment.data.numpy()

    def viz_data(self, plt, num_seqs=20, bound=15, fs=6, colors=['#AA3377', '#EE7733', '#0077BB', '#009988', '#555555', '#999933']):
        for s in range(num_seqs):
            ob, precision, mean, assignment = self.sim_one_gmm()
            fig, ax = plt.subplots(figsize=(fs, fs))
            labels = assignment.argmax(-1)
            for k in range(self.K):
                cov_k = np.diag(1. / precision[k])
                ob_k = ob[np.where(labels == k)]
                ax.scatter(ob_k[:, 0], ob_k[:, 1], c=colors[k], zorder=3)
                plot_cov_ellipse(cov=cov_k, pos=mean[k], nstd=2, color=colors[k], ax=ax, alpha=0.3, zorder=3)
            ax.set_ylim([-bound, bound])
            ax.set_xlim([-bound, bound])
            ax.set_xticks([])
            ax.set_yticks([])

    def sim_save_data(self, num_seqs, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        OB = np.zeros((num_seqs, self.N, self.D))
        ASSIGNMENT = np.zeros((num_seqs, self.N, self.K))
        # MEAN = np.zeros((num_seqs, self.K, self.D))
        # SIGMA = np.zeros((num_seqs, self.K, self.D))
        for s in range(num_seqs):
            ob, precision, mean, assignment = self.sim_one_gmm()
            OB[s] = ob
            ASSIGNMENT[s] = assignment
        print('saving to %s' % os.path.abspath(data_path))
        np.save(data_path + 'ob', OB)
        np.save(data_path + 'assignment', ASSIGNMENT)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('GMM_DATA')
    parser.add_argument('--num_instances', default=10000, type=int)
    parser.add_argument('--data_path', default='../../data/gmm/')
    parser.add_argument('--N', default=60, help='number of points in one GMM instance')
    parser.add_argument('--K', default=3, help='number of clusters in one GMM instance')
    parser.add_argument('--D', default=2, help='dimension of data points')
    parser.add_argument('--alpha', default=2.0, help='alpha parameter of the normal-gamma prior')
    parser.add_argument('--beta', default=2.0, help='beta parameter of the normal-gamma prior')
    parser.add_argument('--mu', default=0.0, help='mu parameter of the normal-gamma prior')
    parser.add_argument('--nu', default=0.1, help='nu/lambda parameter of the normal-gamma prior')
    args = parser.parse_args()
    simulator = Sim_GMM(args.N, args.K, args.D, args.alpha, args.beta, args.mu, args.nu)
    simulator.sim_save_data(args.num_instances, args.data_path)

    # simulator = Sim_GMM(N=60, K=3, D=2, alpha=2.0, beta=2.0, mu=0.0, nu=0.1)
    # simulator.sim_save_data(num_instances=10000, data_path='../../data/gmm/',)
