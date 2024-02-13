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
    def __init__(self, dim_in):
        #Define Geometry
        self.dim_in = dim_in
        self.ub = np.array([])
        self.lb = np.array([])
        for i in range(dim_in):
            self.ub = np.append(self.ub, 1)
            self.lb = np.append(self.lb, 0)

        #Data Hyperparameters
        self.N_b = 8000 # Boundary points
        self.N_c = 10000 # Domain points

    #Construct the geometry
    def __call__(self):
        #Boundary points
        bc_x_left = self.lb + (self.ub - self.lb) * lhs(self.dim_in, self.N_b)
        for i in range(self.dim_in):
            dim_range = int(self.N_b/self.dim_in)
            bc_x_left[i*dim_range:(i+1)*dim_range, i] = np.zeros(dim_range)
        bc_x_left[self.dim_in*dim_range::, -1] = np.zeros(bc_x_left[self.dim_in*dim_range::, -1].shape)
        bc_x_right = self.lb + (self.ub - self.lb) * lhs(self.dim_in, self.N_b)
        for i in range(self.dim_in):
            dim_range = int(self.N_b/self.dim_in)
            bc_x_right[i*dim_range:(i+1)*dim_range, i] = np.ones(dim_range)
        bc_x_right[self.dim_in*dim_range::, -1] = np.ones(bc_x_right[self.dim_in*dim_range::, -1].shape)
        bc_x = np.concatenate((bc_x_left, bc_x_right), axis=0)
        bc_u = np.sin(np.pi / 2 * bc_x).sum(axis=1).reshape(-1, 1)
        
        #Domain points
        col_x = self.lb + (self.ub - self.lb) * lhs(self.dim_in, self.N_c)
        
        #Convert to tensors
        bc_x = torch.tensor(bc_x, dtype=torch.float64).to(device)
        bc_u = torch.tensor(bc_u, dtype=torch.float64).to(device)
        col_x = torch.tensor(col_x, dtype=torch.float64).to(device)

        return bc_x, bc_u, col_x
    
    def __len__(self):
        print("INFO: data size of the geometry: ")
        print(f'Initial:{self.N_i}; Boundary:{self.N_b*2}; Domain{self.N_c}')
        print(f'Total: {self.N_i + self.N_c}')
        return self.N_i + self.N_c

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
    dim_in = 5
    data = Geo_Dataset(dim_in)
    col_x = data()

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