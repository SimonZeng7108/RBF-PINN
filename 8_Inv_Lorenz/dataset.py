import numpy as np
from Utils import sample_Gen, plot_XYZ_2D, plot_XYT_3D, plot_XYTZ_3D
from pyDOE import lhs
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
np.random.seed(1234)

class Geo_Dataset:
    """Construct the geometry dataset"""
    def __init__(self):
        #Define timestep
        self.initial_pos = np.array([0.0, 1.0, 1.05])
        # self.initial_pos = np.array([-8.0, 7.0, 27.0])
        self.dt = 0.001
        self.t_min = 0.0
        # self.t_max = 50.0
        self.t_max = 3.0        
        self.ub = np.array([self.t_max])
        self.lb = np.array([self.t_min])
        #Data Hyperparameters
        data_percent = 0.01 #use percentage of data
        self.N_s = int(data_percent * (self.t_max/self.dt)) # Sample training points
        self.N_c = 400 # Sample collocation points

    #Construct the geometry
    def __call__(self):
        #generate some data points
        data_t = np.arange(0, self.t_max+self.dt, self.dt).reshape(-1, 1)
        data_xyz = np.empty((int(self.t_max/self.dt) + 1, 3))  # Need one more for the initial values
        data_xyz[0] = self.initial_pos  # Set initial values
        for i in range(int(self.t_max/self.dt)):
            data_xyz[i + 1] = data_xyz[i] + self.lorenz(data_xyz[i]) * self.dt
        data_txyz = np.hstack((data_t, data_xyz))

        print('data_txyz', data_txyz.shape)



        #Initial condition
        ic_t = data_txyz[0, 0:1].reshape(-1, 1)
        ic_xyz = data_txyz[0, 1:4].reshape(-1, 3)

        #sample points, with value of XYZ   
        sample_txyz = data_txyz[np.sort(np.random.choice(data_txyz.shape[0], self.N_s, replace=False)), :]
        sample_t = sample_txyz[:, 0:1]
        sample_xyz = sample_txyz[:, 1:4]
        print('sample_txyz', sample_txyz.shape)

        # def gen_traindata():
        #     data = np.load("lorenz_data.npz")
        #     return data["t"], data["y"]
        # sample_t, sample_xyz = gen_traindata()

        #Collocation points
        # col_t = data_txyz[np.sort(np.random.choice(data_txyz.shape[0], self.N_c, replace=False)), 0:1]
        col_t = self.lb + (self.ub - self.lb) * lhs(1, self.N_c)
        col_t = np.sort(col_t, axis=0)
        print('col_t', col_t.shape)
        # assert 1==2

        #Convert to tensors
        ic_t = torch.tensor(ic_t, dtype=torch.float64).to(device)
        ic_xyz = torch.tensor(ic_xyz, dtype=torch.float64).to(device)
        sample_t = torch.tensor(sample_t, dtype=torch.float64).to(device)
        sample_xyz = torch.tensor(sample_xyz, dtype=torch.float64).to(device)
        col_t = torch.tensor(col_t, dtype=torch.float64).to(device)



        return ic_t, ic_xyz, sample_t, sample_xyz, col_t
    
    def lorenz(self, xyz, *, a=10, r=15, b=8/3):
        #a=10, r=28(or 15 for disk like shape), b=8/3
        """
        Parameters
        ----------
        xyz : array-like, shape (3,)
        Point of interest in three-dimensional space.
        s, r, b : float
        Parameters defining the Lorenz attractor.

        Returns
        -------
        xyz_dot : array, shape (3,)
        Values of the Lorenz attractor's partial derivatives at *xyz*.
        """
        x, y, z = xyz
        x_dot = a*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot])


    def __len__(self):
        print("INFO: data size of the geometry: ")
        print(f'Data: {self.N_c}')
        return self.N_c

class Col_Data(Dataset):
    """Create Torch type dataset for Collocations points with no velocity data"""
    def __init__(self, col_xy):
        self.col_xy = col_xy

    def __getitem__(self, idx):
        return self.col_xy[idx]

    def __len__(self):
        return self.col_xy.shape[0]
    
class BC_Data(Dataset):
    """Create Torch type dataset for Boundary points with velocity data"""
    def __init__(self, bc_xy, bc_uv):
        self.bc_xy = bc_xy
        self.bc_uv = bc_uv

    def __getitem__(self, idx):
        return self.bc_xy[idx], self.bc_uv[idx]

    def __len__(self):
        return self.bc_xy.shape[0]
    

if __name__ == "__main__":
    data = Geo_Dataset()
    ic_xt, ic_u, bc_xt, bc_u, col_xt = data()

    col_loader = DataLoader(Col_Data(col_xt), batch_size=1000, shuffle=False)
    # col_loader = DataLoader(Col_Data(col_xy), batch_size=1000, shuffle=False)
    bc_loader = DataLoader(BC_Data(bc_xt, bc_u), batch_size=1000, shuffle=False)
    ic_loader = DataLoader(BC_Data(ic_xt, ic_u), batch_size=1000, shuffle=False)

    print('col', len(Col_Data(col_xt)))
    print('bc', len(BC_Data(bc_xt, bc_u)))

    for col_data, bc_data in zip(col_loader, bc_loader):
        print(col_data.shape)
        bc_data, bc_uv = bc_data[0], bc_data[1]
        print(bc_data.shape)
        # plot_XYZ_2D(colxy[:, 0], colxy[:, 1], show = True)



    for i, (col_xy) in enumerate(col_loader):
        print(col_xy.shape)
        colxy = col_xy.cpu().numpy()
        plot_XYZ_2D(colxy[:, 0], colxy[:, 1], show = True)


    for i, (bc_xy, bc_uv) in enumerate(bc_loader):
        print(bc_xy.shape)
        bcxy = bc_xy.cpu().numpy()
        bcuv = bc_uv.cpu().numpy()
        plot_XYZ_2D(bcxy[:, 0], bcxy[:, 1], bcuv[:, 0], show = True)
    
    for i, (outlet_xy) in enumerate(ic_loader):
        print(outlet_xy.shape)
        outletxy = outlet_xy.cpu().numpy()
        plot_XYZ_2D(outletxy[:, 0], outletxy[:, 1], show = True)