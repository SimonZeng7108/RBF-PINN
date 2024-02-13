from main import dim_in, dim_out, dataset

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
model = RBF_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layer_centres=hidden_layer_centres, ub=dataset.ub, lb=dataset.lb).to(device).to(torch.float64)

model.load_state_dict(torch.load("./logs/models/2024-01-02-16-09-46_2d_steady_ns_rbf128_100X5/epoch_34000.pt"))


datContent = [i.strip().split() for i in open("./data/ns_0_obstacle.dat", encoding="utf8").readlines()]
dataarray = np.array(datContent[9::], dtype=np.float32)
x = dataarray[:, 0].reshape(-1, 1)
y = dataarray[:, 1].reshape(-1, 1)
u = dataarray[:, 2].reshape(-1, 1)
v = dataarray[:, 3].reshape(-1, 1)
p = dataarray[:, 4].reshape(-1, 1)

data = [u, v, p]
label = ["u", "v", "p"]
c = 131
plt.figure(figsize=(5*3, 5))
for i in range(3):
    sub = plt.subplot(c)
    plt.scatter(x, y, c=data[i], cmap='jet',vmin=-1, vmax=1, marker = '.')
    plt.colorbar(shrink=0.5)
    sub.set_title(label[i])
    sub.set_aspect('equal')
    plt.tight_layout()
    c+=1
plt.savefig("./results/ns2d_baseline.png", dpi = 500)




xy = np.concatenate([x, y], axis=1)
xy = torch.tensor(xy, dtype=torch.float64).to(device)

with torch.no_grad():
    out = model(xy)
    u_pred, v_pred, p_pred = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    u_pred = u_pred.cpu().numpy()
    v_pred = v_pred.cpu().numpy()
    p_pred = p_pred.cpu().numpy()


data = [u_pred, v_pred, p_pred]
label = ["u_pred", "v_pred", "p_pred"]
c = 131
plt.figure(figsize=(5*3, 5))
for i in range(3):
    sub = plt.subplot(c)
    plt.scatter(x, y, c=data[i], cmap='jet',vmin=-1, vmax=1, marker = '.')
    plt.colorbar(shrink=0.5)
    sub.set_title(label[i])
    sub.set_aspect('equal')
    plt.tight_layout()
    c+=1
plt.savefig("./results/ns2d_prediction.png", dpi = 500)
plt.show()

L2RE_u = (((u_pred- u)**2).sum()/((u)**2).sum())**0.5
L2RE_v = (((v_pred- v)**2).sum()/((v)**2).sum())**0.5
L2RE_p = (((p_pred- p)**2).sum()/((p)**2).sum())**0.5
ave_L2 = (L2RE_u + L2RE_v + L2RE_p)/3
print("L2RE_u: ", L2RE_u)
print("L2RE_v: ", L2RE_v)
print("L2RE_p: ", L2RE_p)
print("ave_L2: ", ave_L2)