import numpy as np
from Utils import sample_Gen, plot_XYZ_2D, plot_XYT_3D, plot_XYTZ_3D
from pyDOE import lhs
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Geo_Dataset:
    """Construct the geometry dataset"""
    def __init__(self):
        #Define Geometry
        self.x_min = -1.0 #wall left
        self.x_max = 1.0 #wall right
        self.t_min = 0.0 #start 
        self.t_max = 1.0 #end
        self.ub = np.array([self.x_max, self.t_max])
        self.lb = np.array([self.x_min, self.t_min])

        #Data Hyperparameters
        self.N_i = 1000  # Initial Condition
        self.N_b = 1000  # Boundary Condition
        self.N_c = 20000 # Domain points

    #Construct the geometry
    def __call__(self):
        #Initial condition
        ic_x = sample_Gen(self.x_min, self.x_max, self.N_i, True)
        ic_t = np.zeros((self.N_i, 1))
        ic_xt = np.concatenate([ic_x, ic_t], axis=1)
        ic_u = np.sin(np.pi * ic_x)  #Dirichlet condition

        #Boundary condition
        bc_left_x = np.ones((self.N_b, 1)) * self.x_min
        bc_left_t = sample_Gen(self.t_min, self.t_max, self.N_b, True)
        bc_right_x = np.ones((self.N_b, 1)) * self.x_max
        bc_right_t = sample_Gen(self.t_min, self.t_max, self.N_b, True)
        bc_left_xt = np.concatenate([bc_left_x, bc_left_t], axis=1)
        bc_right_xt = np.concatenate([bc_right_x, bc_right_t], axis=1)
        bc_xt = np.concatenate([bc_left_xt, bc_right_xt], axis=0)
        bc_left_u = np.zeros((self.N_b, 1))
        bc_right_u = np.zeros((self.N_b, 1))
        bc_u = np.concatenate([bc_left_u, bc_right_u], axis=0)

        #Domain points
        col_xt = self.lb + (self.ub - self.lb) * lhs(2, self.N_c)


        #Convert to tensors
        ic_xt = torch.tensor(ic_xt, dtype=torch.float64).to(device)
        ic_u = torch.tensor(ic_u, dtype=torch.float64).to(device)
        bc_xt = torch.tensor(bc_xt, dtype=torch.float64).to(device)
        bc_u = torch.tensor(bc_u, dtype=torch.float64).to(device)
        col_xt = torch.tensor(col_xt, dtype=torch.float64).to(device)


        return ic_xt, ic_u, bc_xt, bc_u, col_xt
    
    def __len__(self):
        print("INFO: data size of the geometry: ")
        print(f'Initial:{self.N_i}; Boundary:{self.N_b*2}; Domain{self.N_c}')
        print(f'Total: {self.N_i + self.N_b*2 + self.N_c}')
        return self.N_i + self.N_b*2 + self.N_c

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