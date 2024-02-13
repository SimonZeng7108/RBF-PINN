from main import dim_in, dim_out, dataset
from Utils import plot_XYZ_2D
from network import  DNN_custom, RBF_DNN
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_min = dataset.x_min
x_max = dataset.x_max
y_min = dataset.t_min
y_max = dataset.t_max


hidden_layer_centres = [128, 100, 100, 100, 100, 100] # if customised layer 
Pol_Feat = 10 #Numbe of polynomial features
model = RBF_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layer_centres=hidden_layer_centres, ub=dataset.ub, lb=dataset.lb).to(device).to(torch.float64)
model.load_state_dict(torch.load("./logs/models/2024-01-02-16-10-01_1d_burgers_rbf128_100X5/epoch_34000.pt"))


def gen_testdata():
    data = np.load("data/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    xt = np.vstack((np.ravel(xx), np.ravel(tt))).T
    u_baseline = exact.flatten()[:, None]
    return xt, u_baseline

xt, u_baseline = gen_testdata()

plt.figure(figsize=(5, 5))
plt.scatter(xt[:, 0:1], xt[:, 1:2], c=u_baseline, cmap='jet',vmin=-1, vmax=1, marker = '.')
plt.colorbar(shrink=0.5)
plt.tight_layout()
plt.axis('scaled')
plt.savefig("./results/burgers_baseline.png", dpi= 500)
# plt.show()



xt = torch.tensor(xt, dtype=torch.float64).to(device)


with torch.no_grad():
    out = model(xt)
    u = out[:, 0:1]
    u = u.cpu().numpy()


xt = xt.cpu().numpy()

plt.figure(figsize=(5, 5))
plt.scatter(xt[:, 0:1], xt[:, 1:2], c=u, cmap='jet',vmin=-1, vmax=1, marker = '.')
plt.colorbar(shrink=0.5)
plt.tight_layout()
plt.axis('scaled')
plt.savefig("./results/burgers_prediction.png", dpi= 500)
plt.show()

L2RE = (((u - u_baseline)**2).sum()/((u_baseline)**2).sum())**0.5
print(L2RE)
