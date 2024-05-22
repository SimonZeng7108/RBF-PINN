from main import dim_in, dim_out, dataset
from Utils import plot_XYZ_2D
from network import DNN_custom,RBF_DNN
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_min = dataset.x_min
x_max = dataset.x_max
t_min = dataset.t_min
t_max = dataset.t_max


hidden_layer_centres = [128, 100, 100, 100, 100, 100]
Pol_Feat = 5
model =RBF_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layer_centres=hidden_layer_centres, ub=dataset.ub, lb=dataset.lb, order_pol=Pol_Feat).to(device).to(torch.float64)

model.load_state_dict(torch.load("./logs/models/2024-01-02-16-09-18_1d_wave_DNN_100X5/epoch_34000.pt"))


x = np.arange(x_min, x_max, 0.01)
t = np.arange(t_min, t_max, 0.01)
X, T = np.meshgrid(x, t)
x = X.reshape(-1, 1)
t = T.reshape(-1, 1)
xt = np.concatenate([x, t], axis=1)


#analytical solution
def analytical(x ,t):

    return np.sin(np.pi*x)*np.cos(2*np.pi*t) + 1/2*np.sin(4*np.pi*x)*np.cos(8*np.pi*t)

analytical_u = analytical(xt[:,0], xt[:,1]).reshape(-1, 1)
plt.figure(figsize=(5, 5))
plt.scatter(x, t, c=analytical_u, cmap='jet',vmin=-1, vmax=1, marker = '.')
plt.colorbar(shrink=0.5)
plt.tight_layout()
plt.axis('scaled')
plt.savefig("./results/WAVE_baseline.png", dpi= 500)
# plt.show()





xt = torch.tensor(xt, dtype=torch.float64).to(device)
with torch.no_grad():
    out = model(xt)
    u= out[:, 0:1]
    u = u.cpu().numpy()
xt = xt.cpu().numpy()
plt.figure(figsize=(5, 5))
plt.scatter(x, t, c=u, cmap='jet',vmin=-1, vmax=1, marker = '.')
plt.colorbar(shrink=0.5)
plt.tight_layout()
plt.axis('scaled')
plt.savefig("./results/WAVE_pred_weighted.png", dpi= 500)
plt.show()



L2RE = (((u - analytical_u)**2).sum()/((analytical_u)**2).sum())**0.5
print(L2RE)
