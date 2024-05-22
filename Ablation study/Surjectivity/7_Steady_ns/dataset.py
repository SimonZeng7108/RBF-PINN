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
        self.x_min = 0.0 #wall left
        self.x_max = 4.0 #wall right
        self.y_min = 0.0 #start 
        self.y_max = 2.0 #end
        self.x_mid = 2.0 #horizontal line
        self.y_mid = 1.0 #vertical line
        self.ub = np.array([self.x_max, self.y_max])
        self.lb = np.array([self.x_min, self.y_min])




        #Data Hyperparameters
        self.N_b = 1000  # Boundary Condition
        self.N_c = 20000 # Domain points

    #Construct the geometry
    def __call__(self):
        #Boundary condition
        bc_left_x = np.ones((self.N_b, 1)) * self.x_min
        bc_left_y = sample_Gen(self.y_min, self.y_mid, self.N_b, True)
        bc_left_xy = np.concatenate([bc_left_x, bc_left_y], axis=1)
        bc_top_x = sample_Gen(self.x_mid, self.x_max, self.N_b, True)
        bc_top_y = np.ones((self.N_b, 1)) * self.y_max
        bc_top_xy = np.concatenate([bc_top_x, bc_top_y], axis=1)
        bc_bot_x = sample_Gen(self.x_min, self.x_max, self.N_b, True)
        bc_bot_y = np.ones((self.N_b, 1)) * self.y_min
        bc_bot_xy = np.concatenate([bc_bot_x, bc_bot_y], axis=1)
        bc_horizontal_x = sample_Gen(self.x_min, self.x_mid, int(self.N_b/2), True)
        bc_horizontal_y = np.ones((int(self.N_b/2), 1)) * self.y_mid
        bc_horizontal_xy = np.concatenate([bc_horizontal_x, bc_horizontal_y], axis=1)
        bc_vertical_x = np.ones((int(self.N_b/2), 1)) * self.x_mid
        bc_vertical_y = sample_Gen(self.y_mid, self.y_max, int(self.N_b/2), True)
        bc_vertical_xy = np.concatenate([bc_vertical_x, bc_vertical_y], axis=1)
        bc_xy = np.concatenate([bc_left_xy, bc_top_xy, bc_bot_xy, bc_horizontal_xy, bc_vertical_xy], axis=0)

        bc_left_u = 4 * bc_left_y * (1 - bc_left_y)
        bc_left_v = np.zeros((self.N_b, 1))
        bc_top_u = np.zeros((self.N_b, 1))
        bc_top_v = np.zeros((self.N_b, 1))
        bc_bot_u = np.zeros((self.N_b, 1))
        bc_bot_v = np.zeros((self.N_b, 1))
        bc_horizontal_u = np.zeros((int(self.N_b/2), 1))
        bc_horizontal_v = np.zeros((int(self.N_b/2), 1))
        bc_vertical_u = np.zeros((int(self.N_b/2), 1))
        bc_vertical_v = np.zeros((int(self.N_b/2), 1))
        bc_u = np.concatenate([bc_left_u, bc_top_u, bc_bot_u, bc_horizontal_u, bc_vertical_u], axis=0)
        bc_v = np.concatenate([bc_left_v, bc_top_v, bc_bot_v, bc_horizontal_v, bc_vertical_v], axis=0)
        bc_uv = np.concatenate([bc_u, bc_v], axis=1)
        #outlet
        outlet_x = np.ones((self.N_b, 1)) * self.x_max
        outlet_y = sample_Gen(self.y_min, self.y_max, self.N_b, True)
        outlet_xy = np.concatenate([outlet_x, outlet_y], axis=1)
        outlet_p = np.zeros((self.N_b, 1))


        #Domain points
        col_xy = self.lb + (self.ub - self.lb) * lhs(2, self.N_c)
        # remove collocation points at the bend
        mask = np.bitwise_not((col_xy[:, 0] <=2) & (col_xy[:, 1] >= 1))
        col_xy = col_xy[mask].reshape(-1, 2)


        #Convert to tensors
        bc_xy = torch.tensor(bc_xy, dtype=torch.float64).to(device)
        bc_uv = torch.tensor(bc_uv, dtype=torch.float64).to(device)
        outlet_xy = torch.tensor(outlet_xy, dtype=torch.float64).to(device)
        outlet_p = torch.tensor(outlet_p, dtype=torch.float64).to(device)
        col_xy = torch.tensor(col_xy, dtype=torch.float64).to(device)


        return bc_xy, bc_uv, outlet_xy, outlet_p, col_xy
    
    def __len__(self):
        print("INFO: data size of the geometry: ")
        print(f'Boundary:{self.N_b*4}; Outlet:{self.N_b}; Domain{self.N_c}')
        print(f'Total: {self.N_b*5 + self.N_c}')
        return self.N_b*5 + self.N_c

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
    bc_xt, bc_u, col_xt = data()

    col_loader = DataLoader(Col_Data(col_xt), batch_size=1000, shuffle=False)
    # col_loader = DataLoader(Col_Data(col_xy), batch_size=1000, shuffle=False)
    bc_loader = DataLoader(BC_Data(bc_xt, bc_u), batch_size=1000, shuffle=False)


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
