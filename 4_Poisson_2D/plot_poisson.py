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
y_min = dataset.y_min
y_max = dataset.y_max

hidden_layer_centres = [128, 100, 100, 100, 100, 100] # if customised layer 
Pol_Feat = 0 #Numbe of polynomial features

model = RBF_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layer_centres=hidden_layer_centres, ub=dataset.ub, lb=dataset.lb, order_pol=Pol_Feat).to(device).to(torch.float64)

model.load_state_dict(torch.load("./logs/models/2024-01-02-16-10-01_2d_poisson_rbf128_100X5/epoch_34000.pt"))


datContent = [i.strip().split() for i in open("./data/poisson_classic.dat", encoding="utf8").readlines()]
dataarray = np.array(datContent[0::], dtype=np.float32)
x = dataarray[:, 0].reshape(-1, 1)
y = dataarray[:, 1].reshape(-1, 1)
u_baseline = dataarray[:, 2].reshape(-1, 1)

plt.figure(figsize=(5, 5))
plt.scatter(x, y, c=u_baseline, cmap='jet',vmin=0, vmax=1, marker = '.')
plt.colorbar(shrink=0.5)
plt.tight_layout()
plt.axis('scaled')
plt.savefig("./results/poisson_baseline.png", dpi= 500)
plt.show()



xy = np.concatenate([x, y], axis=1)
xy = torch.tensor(xy, dtype=torch.float64).to(device)


with torch.no_grad():
    out = model(xy)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    u = u.cpu().numpy()



print(u.shape)
print(u_baseline.shape)


L2RE = (((u - u_baseline)**2).sum()/((u_baseline)**2).sum())**0.5
print(L2RE)


# plot_XYZ_2D(x, y, u, show = True)
assert 1==2






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