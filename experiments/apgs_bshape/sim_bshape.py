import os
import gzip
import math
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.distributions.uniform import Uniform
from torch.nn.functional import affine_grid, grid_sample


"""
==========
simulate bouncing mnist using the training dataset in mnist
==========
"""
class Sim_BShape():
    def __init__(self, timesteps, num_objects, frame_size, object_size, offset, dv, chunk_size):
        '''
        X : coordinates
        V : velocity
        '''
        super(Sim_BShape, self).__init__()
        self.timesteps = timesteps
        self.num_objects = num_objects
        self.frame_size = frame_size
        self.object_size = object_size ## by default
        self.offset = offset
        self.dv = dv
        self.chunk_size = chunk_size ## datasets are dividied into pieces with this number and saved separately
    def plot_square(self, size):
        square = torch.ones((size,size))
        return square

    def plot_cross(self, size):
        cross = torch.zeros((size, size))
        for i in range(size):
            cross[i, i] = 1.0
            cross[i, int(size-1-i)] = 1.0
        return cross

    def plot_ball(self, size):
        if size == 6:
            ball = torch.ones((6,6))
            for i in range(6):
                if i == 0 or i == 5:
                    ball[i, 0] = 0.0
                    ball[i, 1] = 0.0
                    ball[i, 4] = 0.0
                    ball[i, 5] = 0.0
                elif i == 1 or i == 4:
                    ball[i, 0] = 0.0
                    ball[i, 5] = 0.0
                else:
                    pass
        elif size == 8:
            ball = torch.ones((8,8))
            for i in range(8):
                if i == 0 or i == 7: 
                    ball[i, 0] = 0.0
                    ball[i, 1] = 0.0
                    ball[i, 2] = 0.0
                    ball[i, 5] = 0.0
                    ball[i, 6] = 0.0
                    ball[i, 7] = 0.0
                elif i == 1 or i == 6:
                    ball[i, 0] = 0.0
                    ball[i, 1] = 0.0
                    ball[i, 6] = 0.0
                    ball[i, 7] = 0.0
                elif i == 2 or i == 5:
                    ball[i, 0] = 0.0
                    ball[i, 7] = 0.0
                else:
                    pass
        elif size == 10:
            ball = torch.ones((10,10))
            for i in range(10):
                if i == 0 or i == 9: 
                    ball[i, 0] = 0.0
                    ball[i, 1] = 0.0
                    ball[i, 2] = 0.0
                    ball[i, 3] = 0.0
                    ball[i, 6] = 0.0
                    ball[i, 7] = 0.0
                    ball[i, 8] = 0.0
                    ball[i, 9] = 0.0
                elif i == 1 or i == 8:
                    ball[i, 0] = 0.0
                    ball[i, 1] = 0.0
                    ball[i, 2] = 0.0
                    ball[i, 7] = 0.0
                    ball[i, 8] = 0.0
                    ball[i, 9] = 0.0
                elif i == 2 or i == 7:
                    ball[i, 0] = 0.0
                    ball[i, 1] = 0.0
                    ball[i, 8] = 0.0
                    ball[i, 9] = 0.0
                elif i == 3 or i == 6:
                    ball[i, 0] = 0.0
                    ball[i, 9] = 0.0
                else:
                    pass
        else:
            raise ValueError
        return ball

    def sim_trajectory(self, init_xs):
        ''' Generate a random sequence of a MNIST digit '''
        v_norm = Uniform(0, 1).sample() * 2 * math.pi
        #v_norm = torch.ones(1) * 2 * math.pi
        v_y = torch.sin(v_norm).item()
        v_x = torch.cos(v_norm).item()
        V0 = torch.Tensor([v_x, v_y]) * self.dv
        X = torch.zeros((self.timesteps, 2))
        V = torch.zeros((self.timesteps, 2))
        X[0] = init_xs
        V[0] = V0
        for t in range(0, self.timesteps -1):
            X_new = X[t] + V[t]
            V_new = V[t]

            if X_new[0] < -1.0:
                X_new[0] = -1.0 + torch.abs(-1.0 - X_new[0])
                V_new[0] = - V_new[0]
            if X_new[0] > 1.0:
                X_new[0] = 1.0 - torch.abs(X_new[0] - 1.0)
                V_new[0] = - V_new[0]
            if X_new[1] < -1.0:
                X_new[1] = -1.0 + torch.abs(-1.0 - X_new[1])
                V_new[1] = - V_new[1]
            if X_new[1] > 1.0:
                X_new[1] = 1.0 - torch.abs(X_new[1] - 1.0)
                V_new[1] = - V_new[1]
            V[t+1] = V_new
            X[t+1] = X_new
        return X, V

    def sim_trajectories(self, num_tjs, save_flag=False):
        Xs = []
        Vs = []
        x0 = Uniform(-1, 1).sample((num_tjs, 2))
        # a2 = 0.5**2
        # while(True):
        #     if ((x0[0] - x0[1])**2).sum() > a2:
        #         break
        #     x0 = Uniform(-1, 1).sample((num_tjs, 2))
        for i in range(num_tjs):
            x, v = self.sim_trajectory(init_xs=x0[i])
            Xs.append(x.unsqueeze(0))
            Vs.append(v.unsqueeze(0))
        if save_flag:
            np.save('pos', torch.cat(Xs, 0).data.numpy())
            np.save('disp', torch.cat(Vs, 0).data.numpy())
        return torch.cat(Xs, 0), torch.cat(Vs, 0)

    def sim_one_bshape(self, objects):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        s_factor = self.frame_size / (self.object_size+self.offset)
        t_factor = (self.frame_size - (self.object_size+self.offset)) / (self.object_size+self.offset)
        bshape = []
        Xs, Vs = self.sim_trajectories(num_tjs=self.num_objects)
        for k in range(self.num_objects):
            object_k = objects[k]
            S = torch.Tensor([[s_factor, 0], [0, s_factor]]).repeat(self.timesteps, 1, 1)
            Thetas = torch.cat((S, Xs[k].unsqueeze(-1) * t_factor), -1)
            grid = affine_grid(Thetas, torch.Size((self.timesteps, 1, self.frame_size, self.frame_size)), align_corners=False)
            bshape.append(grid_sample(object_k.repeat(self.timesteps, 1, 1).unsqueeze(1), grid, mode='nearest', align_corners=False))
            # TJ.append(Xs[n].unsqueeze(0))
            # Init_V.append(V[0.unsqueeze()])
        bshape = torch.cat(bshape, 1).sum(1).clamp(min=0.0, max=1.0)
        return bshape

    def generate_objects(self):
        """
        randomly create one object
        """
        ind = np.random.randint(2, size=self.num_objects)
#         ind = np.arange(3)
        objects = []
        for i in range(len(ind)):
            if ind[i] == 0:
                ball = self.plot_ball(size=self.object_size)
            elif ind[i] == 1:
                ball = self.plot_square(size=self.object_size)
#             elif ind[i] == 2:
#                 ball = self.plot_cross(size=self.object_size)
            else:
                raise ValueError
            ob = torch.zeros(self.object_size+self.offset, self.object_size+self.offset)

            ob[int(self.offset/2):int(self.object_size+self.offset/2), int(self.offset/2):int(self.object_size+self.offset/2)] = ball
            objects.append(ob[None, :, :])
        objects = torch.cat(objects, 0)
        return objects

    def sim_save_data(self, num_seqs, PATH):
        """
        ==========
        way it saves data:
        if num_seqs <= N, then one round of indexing is enough
        if num_seqs > N, then more than one round is needed
        ==========
        """
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        num_seqs_left = num_seqs
        print('Start to simulate bouncing objects sequences...')
        counter = 1
        while(num_seqs_left > 0):
            time_start = time.time()
            num_this_round = min(self.chunk_size, num_seqs_left)
            seqs = []
            for i in range(num_this_round):
                bshape = self.sim_one_bshape(self.generate_objects())
                seqs.append(bshape.unsqueeze(0))
            seqs = torch.cat(seqs, 0)
            assert seqs.shape == (num_this_round, self.timesteps, self.frame_size, self.frame_size), "ERROR! unexpected chunk shape."
            incremental_PATH = PATH + 'ob-%d' % counter
            np.save(incremental_PATH, seqs)
            counter += 1
            num_seqs_left = max(num_seqs_left - num_this_round, 0)
            time_end = time.time()
            print('(%ds) Simulated %d sequences, saved to \'%s\', %d sequences left.' % ((time_end - time_start), num_this_round, incremental_PATH, num_seqs_left))

    def viz_data(self, num_seqs=5, fs=2):
        num_cols = self.timesteps
        num_rows = num_seqs
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.05, hspace=0.05)
        fig = plt.figure(figsize=(fs * num_cols, fs * num_rows))
        for i in range(num_rows):
            bshape = self.sim_one_bshape(self.generate_objects())
            for j in range(num_cols):
                ax = fig.add_subplot(gs[i, j])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(bshape[j], cmap='gray', vmin=0.0, vmax=1.0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Bouncing Shapes')
    parser.add_argument('--num_instances', default=10000, type=int)
    parser.add_argument('--data_path', default='../../data/bshape/')
    parser.add_argument('--timesteps', default=10, help='number of video frames in one video')
    parser.add_argument('--num_objects', default=3, help='number of objects in one video')
    parser.add_argument('--dv', default=0.3, help='constant velocity of the digits')
    parser.add_argument('--frame_size', default=40, help='squared size of the canvas')
    parser.add_argument('--object_size', default=6)
    parser.add_argument('--offset', default=4)
    parser.add_argument('--chunk_size', default=1000, type=int, help='number of sqeuences that are stored in one single file (for the purpose of memory saving)')
    args = parser.parse_args()
    simulator = Sim_BShape(args.timesteps, args.num_objects, args.frame_size, args.object_size, args.offset, args.dv, args.chunk_size)
    simulator.sim_save_data(args.num_instances, args.data_path)
