import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def plotLoss(losses_dict, path, info=["IC", "BC", "PDE"], show = False):
    fig, axes = plt.subplots(1, len(info), sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(len(info)), info):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    if show == True:
        plt.show()
    fig.savefig(path)
    plt.close()


def plot_XYZ_2D(x, y, z = None, i=1, title=None, show=False):
    if z is None:
        z = np.zeros(x.shape)

    fig = plt.figure(i, figsize=(x.max(),y.max()))
    fig.canvas.manager.set_window_title(title)
    plt.scatter(x, y, c=z, cmap='jet',vmin=-1, vmax=1, marker = '.')
    plt.colorbar().ax.set_ylabel('z', rotation=270)
    if show == True:
        plt.show()

def plot_XYT_3D(x, y, t, i=1, title=None, show=False):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, t, s=30, cmap='jet',  marker = '.', alpha=0.8)
    plt.show()

def plot_XYTZ_3D(x, y, t, z=None, i=1, title=None, show=False):
        marker_data = go.Scatter3d(
        x=x, 
        y=y, 
        z=t,  
        marker=dict(
        size=2,
        color=z,                # set color to an array/list of desired values
        colorscale='turbo',   # choose a colorscale
        opacity=0.8),
        mode='markers'
        )
        fig=go.Figure(data=marker_data)
        fig.update_scenes(aspectmode='data')
        if show == True:
            fig.show()

def sample_Gen(start, end, N, if_random):
    if if_random:
        return np.random.uniform(start, end, N).reshape(-1, 1)
    else:
        return np.linspace(start, end, N).reshape(-1, 1)