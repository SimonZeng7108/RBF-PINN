import numpy as np
from Utils import sample_Gen, plot_XYZ_2D, plot_XYT_3D, plot_XYTZ_3D
from pyDOE import lhs
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import os
import scipy.io
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
np.random.seed(1234)

class Geo_Dataset:
    """Construct the geometry dataset"""
    def __init__(self):
        
        self.data = scipy.io.loadmat('./data/Burgers.mat') 
        x = self.data['x']                                   # 256 points between -1 and 1 [256x1]
        t = self.data['t']                                   # 100 time points between 0 and 1 [100x1] 
        usol = self.data['usol']                             # solution of 256x100 grid points
        X, T = np.meshgrid(x,t)                         # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
        self.xt = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        self.u = usol.T.flatten()[:,None]
        self.lb = self.xt[0]
        self.ub = self.xt[-1]

        self.N_f = 5000
        



    #Construct the geometry
    def __call__(self):
        #Generate the geometry
        id_f = np.random.choice(self.xt.shape[0], self.N_f, replace=False)# Randomly chosen points for Interior
        col_xt = self.xt[id_f]
        col_u = self.u[id_f]


        #Convert to tensors
        col_xt = torch.tensor(col_xt, dtype=torch.float64).to(device)
        col_u = torch.tensor(col_u, dtype=torch.float64).to(device)


        return col_xt, col_u
    

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