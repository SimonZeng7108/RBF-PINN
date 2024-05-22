from main import dim_in, dim_out, n_layer, n_node, dataset, hidden_layer_centres
from Utils import plot_XYZ_2D
from network import DNN, DNN_custom,RBF_DNN
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib  as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_min = dataset.x_min
x_max = dataset.x_max
t_min = dataset.t_min
t_max = dataset.t_max



# model = DNN(dim_in=dim_in, dim_out=dim_out, n_layer=n_layer, n_node=n_node, ub=dataset.ub, lb=dataset.lb).to(device).to(torch.float64)
model = RBF_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layer_centres=hidden_layer_centres, ub=dataset.ub, lb=dataset.lb).to(device).to(torch.float64)
model.load_state_dict(torch.load("./logs/models/h4/epoch_34000.pt"))


x = np.arange(x_min, x_max, 0.01)
t = np.arange(t_min, t_max, 0.01)
X, T= np.meshgrid(x, t)
x = X.reshape(-1, 1)
t = T.reshape(-1, 1)


xt = np.concatenate([x, t], axis=1)

#analytical solution
def analytical(x, t):
    u = np.e**(-t)*np.sin(np.pi*x)
    return u.reshape(-1 ,1)
u_analytical = analytical(x, t)
# plt.figure(figsize=(5, 5))
# plt.scatter(x, t, c=u_analytical, cmap='jet',vmin=-1, vmax=1, marker = '.')
# plt.colorbar(shrink=0.5)
# plt.tight_layout()
# plt.axis('scaled')
# plt.savefig("./results/Diffusion_baseline.png", dpi= 500)
# plt.show()


xt = torch.tensor(xt, dtype=torch.float64).to(device)

with torch.no_grad():
    out = model(xt)
    u = out[:, 0:1]
    u = u.cpu().numpy()
plt.subplots(figsize=(5, 5))
plt.scatter(x, t, c=u, cmap='jet',vmin=-1, vmax=1, marker = '.')
plt.axis('scaled')
plt.autoscale(tight=True)
plt.colorbar(shrink = 0.4, ticks=np.linspace(-1, 1, 5))


plt.savefig("./results/diffusion_rbf.png", dpi= 300, bbox_inches='tight', pad_inches=0)
# plt.show()


absolute_e = np.abs(u - u_analytical)
plt.subplots(figsize=(5, 5))
plt.scatter(x, t, c=absolute_e, cmap='jet',vmin=0, vmax=0.0001, marker = '.')
plt.axis('scaled')
plt.autoscale(tight=True)
plt.colorbar(shrink = 0.4, ticks=np.linspace(0, 0.0001, 5))
plt.savefig("./results/diffusion_rbf_ab.png", dpi= 300, bbox_inches='tight', pad_inches=0)
plt.show()



L2RE = (((u - u_analytical)**2).sum()/((u_analytical)**2).sum())**0.5
print(L2RE)




# fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

# data = (u, v, p)


# labels = ["$u(x,y)$", "$v(x,y)$", "$p(x,y)$"]
# for i in range(3):
#     ax = axes[i]
#     im = ax.imshow(
#         data[i], cmap="rainbow", extent=[x_min, x_max, y_min, y_max], origin="lower", vmin = -2, vmax = 2)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="3%", pad="3%")
#     fig.colorbar(im, cax=cax, label=labels[i])
#     ax.set_title(labels[i])
#     ax.set_xlabel("$x$")
#     ax.set_ylabel("$y$")
#     ax.set_aspect("equal")
# fig.tight_layout()
# fig.savefig("./results/solution.png", dpi=500)
# plt.show()