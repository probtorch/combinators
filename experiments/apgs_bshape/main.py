import os
import time
import argparse
import torch
from tqdm import tqdm, trange
from random import shuffle
from combinators import save_models, adam, debug
from combinators.metrics import effective_sample_size

from experiments.apgs_bshape.gibbs import gibbs_sweeps
from experiments.apgs_bshape.models import init_models

def train_apg(num_epochs, lr, batch_size, budget, num_sweeps, timesteps, data_dir, smoketest, **kwargs):
#     torch.autograd.set_detect_anomaly(True)
    device = torch.device(kwargs['device'])
    sample_size = budget // (num_sweeps + 1)
    assert sample_size > 0, 'non-positive sample size =%d' % sample_size
    try:
        mean_shape = torch.load(data_dir + 'mean_shape.pt').to(device)
    except:
        print("in", os.getcwd())
        raise
    data_paths = []
    for file in os.listdir(data_dir+'/video/'):
        if file.endswith('.pt') and \
        'timesteps=%d-objects=%d' % (timesteps, kwargs['num_objects']) in file:
            data_paths.append(os.path.join(data_dir+'/video/', file))
    if len(data_paths) == 0:
        raise ValueError('Empty data path list.')
    model_version = 'apg-timesteps=%d-objects=%d-sweeps=%d-samples=%d' % (timesteps, kwargs['num_objects'], num_sweeps, sample_size)
    models = init_models(mean_shape=mean_shape, **kwargs)
    optimizer = adam(models.values(), lr=lr, betas=(0.9,0.99))
    apg = gibbs_sweeps(models, num_sweeps, timesteps)
    print('Training for ' + model_version)
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    counter = 0 if smoketest[0] else None
    log_file = None if smoketest[0] else open('./results/log-' + model_version + '.txt', 'a+')

    sample_dims, batch_dim = 0, 1
    detached_mean = lambda t: t.detach().mean().cpu().item()
    ebar = trange(num_epochs)

    def cleanup(start, epoch, group, bars, log_file, metrics, num_batches):
        [bar.close() for bar in bars]
        metrics_print = ",  ".join(['%s: %.4f' % (k, v/num_batches) for k, v in metrics.items()])
        end = time.time()
        print("(%ds) Epoch=%d, Group=%d, " % (end - start, epoch+1, group+1) + metrics_print, flush=True,
              **({} if log_file is None else dict(file=log_file)))
        return metrics_print

    for epoch in ebar:
        shuffle(data_paths)
        start = time.time()
        bbar = tqdm(enumerate(data_paths), total=len(data_paths))
        for group, data_path in bbar:
            metrics = {'ess' : 0.0, 'log_p' : 0.0, 'loss' : 0.0}
            data = torch.load(data_path)
            N, T, _, _ = data.shape
            assert T == timesteps, 'Data contain %d timesteps while the corresponding arugment in APG is %d.' % (T, timesteps)
            num_batches = 1 if N <= batch_size else data.shape[0] // batch_size
            seq_indices = torch.randperm(N)
            tbar = trange(num_batches)
            for b in tbar:
                optimizer.zero_grad()
                frames = data[seq_indices[b*batch_size : (b+1)*batch_size]]
                frames_expand = frames.to(device).repeat(sample_size, 1, 1, 1, 1)
                out = apg(c={"frames": frames_expand}, sample_dims=sample_dims, batch_dim=1, reparameterized=False)
                out.loss.backward()
                optimizer.step()

                metrics['ess'] += detached_mean(effective_sample_size(
                        out.lw.detach(), sample_dims=sample_dims
                    )) if num_sweeps > 0 else 0

                metrics['log_p'] += detached_mean(out.trace.log_joint(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=False))
                metrics['loss'] += out.loss.detach().cpu().item()

                if smoketest[0]:
                    counter+=1
                    if counter >= smoketest[1]:
                        cleanup(start, epoch, group, [ebar, bbar, tbar], log_file, metrics, num_batches)
                        return
            save_models(models, 'cp-' + model_version)
            if not smoketest[0]:
                metrics_print = cleanup(start, epoch, group, [], log_file, metrics, num_batches)
                ebar.set_postfix_str(metrics_print)

    if not smoketest[0]:
        log_file.close()
        # return out, frames

def test_gibbs_sweep(budget, num_sweeps, timesteps, data_dir, **kwargs):
    device = torch.device(kwargs['device'])
    sample_size = budget // (num_sweeps+1)
    mean_shape = torch.load(data_dir + 'mean_shape.pt').to(device)
    data_paths = []
    for file in os.listdir(data_dir+'/video/'):
        if file.endswith('.pt'):
            data_paths.append(os.path.join(data_dir+'/video/', file))
    frames = torch.load(data_paths[0])[:,:timesteps]
    frames_expand = frames.to(device).repeat(sample_size, 1, 1, 1, 1)
    models = init_models(mean_shape=mean_shape, **kwargs)
    apg = gibbs_sweeps(models, num_sweeps, timesteps)
    c = {"frames": frames_expand}
    return apg(c, sample_dims=0, batch_dim=1, reparameterized=False), frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Bouncing Shapes')
    # data config
    parser.add_argument('--data_dir', default='./dataset/')
    parser.add_argument('--frame_pixels', default=96, type=int)
    parser.add_argument('--shape_pixels', default=28, type=int)
    parser.add_argument('--timesteps', default=10, type=int)
    parser.add_argument('--num_objects', default=3, type=int)
    # training config
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--budget', default=120, type=int)
    parser.add_argument('--num_sweeps', default=5, type=int)
    # network config
    parser.add_argument('--num_hidden_digit', default=400, type=int)
    parser.add_argument('--num_hidden_coor', default=400, type=int)
    parser.add_argument('--z_where_dim', default=2, type=int)
    parser.add_argument('--z_what_dim', default=10, type=int)
    # test config
    parser.add_argument('--test', default=False, type=bool)
    # CI config
    parser.add_argument('--smoketest', default=False, type=bool)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--seed', default=1, type=int)

    args = parser.parse_args()
    debug.seed(args.seed)

    if args.test:
        out, frames = test_gibbs_sweep(
            budget=args.budget,
            num_sweeps=args.num_sweeps,
            timesteps=args.timesteps,
            data_dir=args.data_dir,
            frame_pixels=args.frame_pixels,
            shape_pixels=args.shape_pixels,
            num_hidden_digit=args.num_hidden_digit,
            num_hidden_coor=args.num_hidden_coor,
            z_where_dim=args.z_where_dim,
            z_what_dim=args.z_what_dim,
            num_objects=args.num_objects,
            device=args.device)
    else:
        train_apg(num_epochs=args.num_epochs,
                  lr=args.lr,
                  batch_size=args.batch_size,
                  budget=args.budget,
                  num_sweeps=args.num_sweeps,
                  timesteps=args.timesteps,
                  data_dir=args.data_dir,
                  frame_pixels=args.frame_pixels,
                  shape_pixels=args.shape_pixels,
                  num_hidden_digit=args.num_hidden_digit,
                  num_hidden_coor=args.num_hidden_coor,
                  z_where_dim=args.z_where_dim,
                  z_what_dim=args.z_what_dim,
                  num_objects=args.num_objects,
                  device=args.device,
                  smoketest=(args.smoketest, args.iterations),
                  )
    print("done!")
