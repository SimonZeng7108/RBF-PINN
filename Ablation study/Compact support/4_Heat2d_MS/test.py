from main import dim_in, dim_out, n_layer, n_node, dataset, hidden_layer_centres
from Utils import plot_XYZ_2D
from network import DNN, DNN_custom,RBF_DNN
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_min = dataset.x_min
x_max = dataset.x_max
y_min = dataset.y_min
y_max = dataset.y_max
t_min = dataset.t_min
t_max = dataset.t_max
# model = DNN(dim_in=dim_in, dim_out=dim_out, n_layer=n_layer, n_node=n_node, ub=dataset.ub, lb=dataset.lb).to(device).to(torch.float64)
model = RBF_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layer_centres=hidden_layer_centres, ub=dataset.ub, lb=dataset.lb).to(device).to(torch.float64)
model.load_state_dict(torch.load("./logs/models/h4/epoch_34000.pt"))


datContent = [i.strip().split() for i in open("./data/heat_multiscale.dat", encoding="utf8").readlines()]
dataarray = np.array(datContent[9::], dtype=np.float32)
x = dataarray[:, 0].reshape(-1, 1)
y = dataarray[:, 1].reshape(-1, 1)
t0 = np.ones_like(x) * 0.0
t1 = np.ones_like(x) * 1.0
t2 = np.ones_like(x) * 2.0
t3 = np.ones_like(x) * 3.0
t4 = np.ones_like(x) * 4.0
t5 = np.ones_like(x) * 5.0
all_t = np.concatenate([t0, t1, t2, t3, t4, t5], axis=1)


#baselion solution
u_t0 = dataarray[:, 2].reshape(-1, 1)
u_t1 = dataarray[:, 12].reshape(-1, 1)
u_t2 = dataarray[:, 22].reshape(-1, 1)
u_t3 = dataarray[:, 32].reshape(-1, 1)
u_t4 = dataarray[:, 42].reshape(-1, 1)
u_t5 = dataarray[:, 52].reshape(-1, 1)
data = [u_t0, u_t1, u_t2, u_t3, u_t4, u_t5]
label = ["u @t0", "u @t1", "u @t2", "u @t3", "u @t4", "u @t5"]
c = 161
plt.figure(figsize=(5*6, 5))
for i in range(6):
    sub = plt.subplot(c)
    plt.scatter(x, y, c=data[i], cmap='jet',vmin=-1, vmax=1, marker = '.')
    plt.colorbar(shrink=0.5)
    sub.set_title(label[i])
    sub.set_aspect('equal')
    plt.tight_layout()
    c+=1
plt.savefig("./results/heatMS_baseline.png", dpi= 100)
# plt.show()



all_u_pred = []
xy = np.concatenate([x, y], axis=1)

for i in range(6):
    xyt = np.concatenate([x, y, all_t[:, i: i+1]], axis=1)
    xyt = torch.tensor(xyt, dtype=torch.float64).to(device)
    with torch.no_grad():
        out = model(xyt)
        u = out[:, 0:1]
        u = u.cpu().numpy()
        all_u_pred.append(u)

xyt = xyt.cpu().numpy()
label = ["u @t0", "u @t1", "u @t2", "u @t3", "u @t4", "u @t5"]
c = 161
plt.figure(figsize=(5*6, 5))
for i in range(6):
    sub = plt.subplot(c)
    plt.scatter(x, y, c=all_u_pred[i], cmap='jet',vmin=-1, vmax=1, marker = '.')
    plt.colorbar(shrink=0.8)
    sub.set_title(label[i])
    sub.set_aspect('equal')
    plt.axis('scaled')
    plt.autoscale(tight=True)
    plt.tight_layout()
    c+=1
plt.savefig("./results/heatMS_RBF.png", dpi= 60, bbox_inches='tight', pad_inches=0)



error = [x - y  for x, y in zip(data, all_u_pred)]
label = ["u @t0", "u @t1", "u @t2", "u @t3", "u @t4", "u @t5"]
c = 161
plt.figure(figsize=(5*6, 5))
for i in range(6):
    sub = plt.subplot(c)
    plt.scatter(x, y, c=error[i], cmap='jet',vmin=0, vmax=0.01, marker = '.')
    plt.colorbar(shrink=0.8)
    sub.set_title(label[i])
    sub.set_aspect('equal')
    plt.axis('scaled')
    plt.autoscale(tight=True)
    plt.tight_layout()
    c+=1
plt.savefig("./results/heatMS_RBF_abs.png", dpi= 60, bbox_inches='tight', pad_inches=0)
plt.show()




L2RE = (((all_u_pred[0] - u_t0)**2).sum()/((u_t0)**2).sum())**0.5
print(L2RE)