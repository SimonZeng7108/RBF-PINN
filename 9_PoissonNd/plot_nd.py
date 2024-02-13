from main import dim_in, dim_out,  dataset
from Utils import plot_XYTZ_3D, plot_XYT_3D
from pyDOE import lhs
from network import  DNN_custom,RBF_DNN
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_layer_centres = [128, 100, 100, 100, 100, 100] # if customised layer 
Pol_Feat = 10 #Numbe of polynomial features
model = RBF_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layer_centres=hidden_layer_centres, ub=dataset.ub, lb=dataset.lb).to(device).to(torch.float64)

model.load_state_dict(torch.load("./logs/models/2023-11-14-10-12-24_1d_ks_rbf128_100X5/epoch_34000.pt"))


def analytical(X):
    u = 0
    for i in range(X.shape[1]):
        u += np.sin(np.pi/2*X[:, i:i+1])
    return u


sample_points = 1000
ub = dataset.ub
lb = dataset.lb
x = lb + (ub - lb) * lhs(dim_in, 10000)
u_analytical = analytical(x).reshape(-1, 1)
plot_XYT_3D(x[:,0], x[:, 1], x[:, 2], c = u_analytical, show=True)

x = torch.tensor(x, dtype=torch.float64).to(device)
with torch.no_grad():
    out = model(x)
    u = out[:, 0:1]
    u = u.cpu().numpy()
x = x.cpu().numpy()
plot_XYT_3D(x[:,0], x[:, 1], x[:, 2], c = u, show=True)

print("L2RE: ", (((u- u_analytical)**2).sum()/((u_analytical)**2).sum())**0.5)
assert 1==2




samples = 10
dim = dim_in
x_list = []
for i in range(dim):
    x_list.append(np.linspace(0, 1, samples).reshape(-1, 1))
x = np.meshgrid(*x_list)
x = np.concatenate([x[i].reshape(-1, 1) for i in range(dim)], axis=1)
u_analytical = analytical(x).reshape(-1, 1)
plot_XYT_3D(x[:,0], x[:, 1], x[:, 2], c = u_analytical, show=True)



x = torch.tensor(x, dtype=torch.float64).to(device)
with torch.no_grad():
    out = model(x)
    u = out[:, 0:1]
    u = u.cpu().numpy()
x = x.cpu().numpy()
plot_XYT_3D(x[:,0], x[:, 1], x[:, 2], c = u, show=True)

print("L2RE: ", (((u- u_analytical)**2).sum()/((u_analytical)**2).sum())**0.5)


assert 1==2























data = []
slices = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
sample = 5

for i in slices:
    for j in slices:
        x1 = np.linspace(0, 1, sample).reshape(-1, 1)
        x2 = np.linspace(0, 1, sample).reshape(-1, 1)
        x3 = np.linspace(0, 1, sample).reshape(-1, 1)
        x4 = np.ones((sample, 1)) * i
        x5 = np.ones((sample, 1)) * j
        x1, x2, x3, x4, x5 = np.meshgrid(x1, x2, x3, x4, x5)
        x1, x2, x3, x4, x5 = x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1), x4.reshape(-1, 1), x5.reshape(-1, 1)
        X = np.concatenate([x1, x2, x3, x4, x5], axis=1)
        u = analytical(X)
        data.append(u.reshape(-1, 1))



fig, axs = plt.subplots(6, 6,figsize=(12,12),subplot_kw=dict(projection='3d'))
axs = axs.flatten()
x4_label = np.repeat(slices, len(slices))
x5_label = np.tile(slices, len(slices))
for i in range(len(data)):
    # ax = fig.add_subplot(6, 6, i+1, projection = '3d')
    im = axs[i].scatter(x1, x2, x3, c = data[i], s=30, cmap='jet',  marker = '.', alpha=0.8,vmin=0, vmax=5)
    axs[i].set_yticklabels([])
    axs[i].set_xticklabels([])
    axs[i].set_zticklabels([])
    axs[i].set_title(f'x4={x4_label[i]}, x5={x5_label[i]}', fontsize = 8)
plt.colorbar(im, ax=axs, shrink = 0.5)
plt.tight_layout()
plt.savefig("./results/poissonnd_pred.png", dpi=500)
# plt.show()



prediciton = []
for i in slices:
    for j in slices:
        x1 = np.linspace(0, 1, sample).reshape(-1, 1)
        x2 = np.linspace(0, 1, sample).reshape(-1, 1)
        x3 = np.linspace(0, 1, sample).reshape(-1, 1)
        x4 = np.ones((sample, 1)) * i
        x5 = np.ones((sample, 1)) * j
        x1, x2, x3, x4, x5 = np.meshgrid(x1, x2, x3, x4, x5)
        x1, x2, x3, x4, x5 = x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1), x4.reshape(-1, 1), x5.reshape(-1, 1)
        X = np.concatenate([x1, x2, x3, x4, x5], axis=1)
        X = torch.tensor(X, dtype=torch.float64).to(device)

        with torch.no_grad():
            out = model(X)
            u = out[:, 0:1]
            u = u.cpu().numpy()
            prediciton.append(u.reshape(-1, 1))

fig, axs = plt.subplots(6, 6,figsize=(12,12),subplot_kw=dict(projection='3d'))
axs = axs.flatten()
x4_label = np.repeat(slices, len(slices))
x5_label = np.tile(slices, len(slices))
for i in range(len(prediciton)):
    # ax = fig.add_subplot(6, 6, i+1, projection = '3d')
    im = axs[i].scatter(x1, x2, x3, c = prediciton[i], s=30, cmap='jet',  marker = '.', alpha=0.8,vmin=0, vmax=2)
    axs[i].set_yticklabels([])
    axs[i].set_xticklabels([])
    axs[i].set_zticklabels([])
    axs[i].set_title(f'x4={x4_label[i]}, x5={x5_label[i]}', fontsize = 8)
plt.colorbar(im, ax=axs, shrink = 0.5)
plt.tight_layout()
plt.savefig("./results/poissonnd_pred.png", dpi=500)
plt.show()

sumL2RE = 0
for i in range(len(prediciton)):
    print("L2RE: ", (((prediciton[i]- data[i])**2).sum()/((data[i])**2).sum())**0.5)
    sumL2RE += (((prediciton[i]- data[i])**2).sum()/((data[i])**2).sum())**0.5
ave_L2 = sumL2RE/len(prediciton)
print("ave_L2: ", ave_L2)