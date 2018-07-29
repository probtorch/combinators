import numpy as np
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import Range1d
from bokeh.io import push_notebook, show, output_notebook

def step(state, dt, box_bound, signal_noise_ratio):
    xy_new = state[ :2] + state[2: ] * dt + np.random.randn(2) * signal_noise_ratio
    while(True):
        if xy_new[0] < box_bound[0]:
            x_over = abs(box_bound[0] - xy_new[0])
            xy_new[0] = box_bound[0] + x_over
            state[2] *= -1
        elif xy_new[0] > box_bound[1]:
            x_over = abs(box_bound[1] - xy_new[0])
            xy_new[0] = box_bound[1] - x_over
            state[2] *= -1
        elif xy_new[1] < box_bound[2]:
            y_over = abs(box_bound[2] - xy_new[1])
            xy_new[1] = box_bound[2] + y_over
            state[3] *= -1
        elif xy_new[1] > box_bound[3]:
            y_over = abs(box_bound[3] - xy_new[1])
            xy_new[1] = box_bound[3] - y_over
            state[3] *= -1
        else:
            state[:2] = xy_new
            break
    return state

def compute_hidden(velocity):
    if velocity[0] > 0 and velocity[1] > 0:
        return 0
    if velocity[0] > 0 and velocity[1] < 0:
        return 1
    if velocity[0] < 0 and velocity[1] < 0:
        return 2
    if velocity[0] < 0 and velocity[1] > 0:
        return 3

def transition(A, old_hidden, new_hidden):
    A[old_hidden, new_hidden] += 1
    return A

def intialization(T, num_series, Boundary):
    x0 = Boundary * np.random.random(num_series) * np.random.choice([-1,1], size=num_series)
    y0 = Boundary * np.random.random(num_series) * np.random.choice([-1,1], size=num_series)
    init_v = np.random.random((num_series, 2)) 
    v_norm = ((init_v **2 ).sum(1)) ** 0.5 ## compute norm for each initial velocity
    init_v = init_v / v_norm[:, None] ## to make the velocity lying on the unit circle
    init_v_rand_dir = init_v * np.random.choice([-1,1], size=(num_series,2))
    return x0, y0, init_v, init_v_rand_dir

def generate_data(T, dt, init_state, Boundary, signal_noise_ratio):
    A = np.zeros((4,4))
    STATE = np.zeros((T+1, 4)) # since I want to make the displacement of length T, I need the coordinates of length T+1
    box_bound = np.array([-1, 1, -1, 1]) * Boundary
#     plot = figure(plot_width=300, plot_height=300)
#     plot.x_range = Range1d(box_bound[0], box_bound[1])
#     plot.y_range = Range1d(box_bound[2], box_bound[3])
#     c = plot.circle(x=[init_state[0]], y=[init_state[1]])
#     target = show(plot, notebook_handle=True)
    state = init_state
    STATE[0] = state
#     old_hidden = init_hidden(init_state[2:])
    for i in range(T):
        state = step(state, dt, box_bound, signal_noise_ratio)
#         print(state[2:])
        STATE[i+1] = state
        if i == 0:
            old_hidden = compute_hidden(state[2:])
        new_hidden = compute_hidden(state[2:])
        A = transition(A, old_hidden, new_hidden)
        old_hidden = new_hidden
#         c.data_source.data['x'] = [state[0]]
#         c.data_source.data['y'] = [state[1]]
#         push_notebook(handle=target)
#         time.sleep(0.2)
    Disp = (STATE[1:] - STATE[:T])[:, :2]

    A = A / A.sum(1)[:, None]
    return STATE, Disp, A