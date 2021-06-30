import os
import math
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.transform import resize
import torch
from torch.distributions.uniform import Uniform
from torch.nn.functional import affine_grid, grid_sample


"""
simulate bouncing shapes.
"""
class Sim_BShape():
    def __init__(self, timesteps, num_objects, frame_size, shape_size, padding_size, blacklist_shapes, shape_dir, dv, chunk_size):
        '''
        X : coordinates
        V : velocity
        '''
        super(Sim_BShape, self).__init__()
        self.timesteps = timesteps
        self.num_objects = num_objects
        self.frame_size = frame_size
        self.shape_size = shape_size
        self.padding_size = padding_size
        self.dv = dv
        self.chunk_size = chunk_size ## datasets are dividied into pieces with this number and saved separately
        self.blacklist_shapes = blacklist_shapes
        self.shapes = []
        self.shape_dir = shape_dir
        
    def load_resize_shapes(self):
        """
        load and resize all shape images.
        """
        shape_paths = []
        for file in os.listdir(self.shape_dir):
            if file.endswith(".png"):
                if file not in self.blacklist_shapes:
                    shape_paths.append(os.path.join(self.shape_dir, file))

        for p in shape_paths:
            img = imageio.imread(p)
            resized_img = resize(img, (self.shape_size, self.shape_size))[:,:,3]
            if self.padding_size > 0 and isinstance(self.padding_size, int):
                padded_size = self.shape_size + 2 * self.padding_size
                padded_img = np.zeros((padded_size, padded_size))
                padded_img[self.padding_size:self.padding_size+self.shape_size, self.padding_size:self.padding_size+self.shape_size] = resized_img
            else:
                padded_img = resized_img
            self.shapes.append(padded_img[None,:,:])
        self.shapes = torch.from_numpy(np.concatenate(self.shapes, 0)).float()
        mean_shape = self.shapes.mean(0)
        torch.save(mean_shape, 'mean_shape.pt')    
            
    def sim_trajectory(self, init_xs):
        ''' Generate a random trajectory '''
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
        a2 = 0.5
        while(True):
            if ((x0[0] - x0[1])**2).sum() > a2 and ((x0[2] - x0[1])**2).sum() > a2 and ((x0[0] - x0[2])**2).sum() > a2:
                break
            x0 = Uniform(-1, 1).sample((num_tjs, 2))
        for i in range(num_tjs):
            x, v = self.sim_trajectory(init_xs=x0[i])
            Xs.append(x.unsqueeze(0))
            Vs.append(v.unsqueeze(0))
        if save_flag:
            np.save('pos', torch.cat(Xs, 0).data.numpy())
            np.save('disp', torch.cat(Vs, 0).data.numpy())
        return torch.cat(Xs, 0), torch.cat(Vs, 0)

    def sim_one_sequence(self, objects):
        '''
        Get random trajectories for the objects and generate a video.
        '''
        patch_size = self.shape_size + self.padding_size * 2
        s_factor = self.frame_size / (patch_size)
        t_factor = (self.frame_size - (patch_size)) / (patch_size)
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
    
    def sim_save_data(self, num_seqs, PATH):
        """
        ==========
        way it saves data:
        if num_seqs <= N, then one round of indexing is enough
        if num_seqs > N, then more than one round is needed
        ==========
        """
        self.load_resize_shapes()
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
                indices = np.random.permutation(len(self.shapes))[:self.num_objects]
                bshape = self.sim_one_sequence(self.shapes[indices])
                seqs.append(bshape.unsqueeze(0))
            seqs = torch.cat(seqs, 0)
            assert seqs.shape == (num_this_round, self.timesteps, self.frame_size, self.frame_size), "ERROR! unexpected chunk shape."
            incremental_PATH = PATH + 'seq-timesteps=%d-objects=%d-p%d' % (self.timesteps, self.num_objects, counter)
            torch.save(seqs, incremental_PATH + '.pt')
            counter += 1
            num_seqs_left = max(num_seqs_left - num_this_round, 0)
            time_end = time.time()
            print('(%ds) Simulated %d sequences, saved to \'%s\', %d sequences left.' % ((time_end - time_start), num_this_round, incremental_PATH, num_seqs_left))

    def viz_data(self, num_seqs=5, fs=1.5):
        self.load_resize_shapes()
        num_cols = self.timesteps
        num_rows = num_seqs
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.05, hspace=0.05)
        fig = plt.figure(figsize=(fs * num_cols, fs * num_rows))
        for i in range(num_rows):
            indices = np.random.permutation(len(self.shapes))[:self.num_objects]
            bshape = self.sim_one_sequence(self.shapes[indices])
            for j in range(num_cols):
                ax = fig.add_subplot(gs[i, j])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(bshape[j], cmap='gray', vmin=0.0, vmax=1.0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Bouncing Shapes')
    parser.add_argument('--num_instances', default=10000, type=int)
    parser.add_argument('--saving_path', default='video/')
    parser.add_argument('--timesteps', default=10, type=int, help='number of video frames in one video')
    parser.add_argument('--num_objects', default=3, type=int, help='number of objects in one video')
    parser.add_argument('--dv', default=0.1, type=float, help='constant velocity of the digits')
    parser.add_argument('--frame_size', default=96, type=int, help='squared size of the canvas')
    parser.add_argument('--shape_size', default=20, type=int, help='size of the shape')
    parser.add_argument('--padding_size', default=4, type=int, help='padding size')
    parser.add_argument('--blacklist_shapes', default='[]')
    parser.add_argument('--shape_dir', default='shapes/')
    parser.add_argument('--chunk_size', default=1000, type=int, help='number of sqeuences that are stored in one single file (for the purpose of memory saving)')
    args = parser.parse_args()
    simulator = Sim_BShape(args.timesteps, args.num_objects, args.frame_size, args.shape_size, args.padding_size, eval(args.blacklist_shapes), args.shape_dir, args.dv, args.chunk_size)
    print('simulating with %d objects, %d timesteps, dv=%.1f' % (args.num_objects, args.timesteps, args.dv))
    simulator.sim_save_data(args.num_instances, args.saving_path)