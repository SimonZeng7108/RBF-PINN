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
        self.x_min = -0.5 #wall left
        self.x_max = 0.5 #wall right
        self.y_min = -0.5 #start 
        self.y_max = 0.5 #end
        self.ub = np.array([self.x_max, self.y_max])
        self.lb = np.array([self.x_min, self.y_min])

        #Data Hyperparameters
        self.N_b = 1000  # Boundary Condition
        self.N_c = 20000 # Domain points

    #Construct the geometry
    def __call__(self):
        #Boundary condition
        bc_left_x = np.ones((self.N_b, 1)) * self.x_min
        bc_left_y = sample_Gen(self.y_min, self.y_max, self.N_b, True)
        bc_left_xy = np.concatenate([bc_left_x, bc_left_y], axis=1)
        bc_right_x = np.ones((self.N_b, 1)) * self.x_max
        bc_right_y = sample_Gen(self.y_min, self.y_max, self.N_b, True)
        bc_right_xy = np.concatenate([bc_right_x, bc_right_y], axis=1)
        bc_top_x = sample_Gen(self.x_min, self.x_max, self.N_b, True)
        bc_top_y = np.ones((self.N_b, 1)) * self.y_max
        bc_top_xy = np.concatenate([bc_top_x, bc_top_y], axis=1)
        bc_bot_x = sample_Gen(self.x_min, self.x_max, self.N_b, True)
        bc_bot_y = np.ones((self.N_b, 1)) * self.y_min
        bc_bot_xy = np.concatenate([bc_bot_x, bc_bot_y], axis=1)
        bc_xy = np.concatenate([bc_left_xy, bc_right_xy, bc_top_xy, bc_bot_xy], axis=0)
        bc_left_u = np.ones((self.N_b, 1))
        bc_right_u = np.ones((self.N_b, 1))
        bc_top_u = np.ones((self.N_b, 1))
        bc_bot_u = np.ones((self.N_b, 1))
        bc_u = np.concatenate([bc_left_u, bc_right_u, bc_top_u, bc_bot_u], axis=0)

        #surface of cylinders, u = 1
        p1 = (0.3, 0.3, 0.1)
        p2 = (0.3, -0.3, 0.1)
        p3 = (-0.3, 0.3, 0.1)
        p4 = (-0.3, -0.3, 0.1)
        theta = sample_Gen(0, 2 * np.pi, self.N_b, False)
        p1_x = (p1[0] + p1[2] * np.cos(theta)).reshape(-1, 1)
        p1_y = (p1[1] + p1[2] * np.sin(theta)).reshape(-1, 1)
        p1_xy = np.concatenate([p1_x, p1_y], axis=1)
        p1_u = np.zeros((self.N_b, 1))
        p2_x = (p2[0] + p2[2] * np.cos(theta)).reshape(-1, 1)
        p2_y = (p2[1] + p2[2] * np.sin(theta)).reshape(-1, 1)
        p2_xy = np.concatenate([p2_x, p2_y], axis=1)
        p2_u = np.zeros((self.N_b, 1))
        p3_x = (p3[0] + p3[2] * np.cos(theta)).reshape(-1, 1)
        p3_y = (p3[1] + p3[2] * np.sin(theta)).reshape(-1, 1)
        p3_xy = np.concatenate([p3_x, p3_y], axis=1)
        p3_u = np.zeros((self.N_b, 1))
        p4_x = (p4[0] + p4[2] * np.cos(theta)).reshape(-1, 1)
        p4_y = (p4[1] + p4[2] * np.sin(theta)).reshape(-1, 1)
        p4_xy = np.concatenate([p4_x, p4_y], axis=1)
        p4_u = np.zeros((self.N_b, 1))
        cyl_xy = np.concatenate([p1_xy, p2_xy, p3_xy, p4_xy], axis=0)
        cyl_u = np.concatenate([p1_u, p2_u, p3_u, p4_u], axis=0)
        bc_xy = np.concatenate([bc_xy, cyl_xy], axis=0)
        bc_u = np.concatenate([bc_u, cyl_u], axis=0)

        #Domain points
        col_xy = self.lb + (self.ub - self.lb) * lhs(2, self.N_c)
        # remove collocation points inside the cylinder
        dst_from_cyl1 = np.sqrt((col_xy[:, 0] - p1[0]) ** 2 + (col_xy[:, 1] - p1[1]) ** 2)
        col_xy = col_xy[dst_from_cyl1 > (p1[2]+0.00001)].reshape(-1, 2)
        dst_from_cyl2 = np.sqrt((col_xy[:, 0] - p2[0]) ** 2 + (col_xy[:, 1] - p2[1]) ** 2)
        col_xy = col_xy[dst_from_cyl2 > (p2[2]+0.00001)].reshape(-1, 2)
        dst_from_cyl3 = np.sqrt((col_xy[:, 0] - p3[0]) ** 2 + (col_xy[:, 1] - p3[1]) ** 2)
        col_xy = col_xy[dst_from_cyl3 > (p3[2]+0.00001)].reshape(-1, 2)
        dst_from_cyl4 = np.sqrt((col_xy[:, 0] - p4[0]) ** 2 + (col_xy[:, 1] - p4[1]) ** 2)
        col_xy = col_xy[dst_from_cyl4 > (p4[2]+0.00001)].reshape(-1, 2)





        #Convert to tensors
        plot_XYTZ_3D(bc_xy[:, 0], bc_xy[:, 1], bc_u[:, 0], bc_u[:, 0], bc_u[:, 0], show = True)
        bc_xy = torch.tensor(bc_xy, dtype=torch.float64).to(device)
        bc_u = torch.tensor(bc_u, dtype=torch.float64).to(device)
        col_xy = torch.tensor(col_xy, dtype=torch.float64).to(device)


        return bc_xy, bc_u, col_xy
    
    def __len__(self):
        print("INFO: data size of the geometry: ")
        print(f'Boundary:{self.N_b*8}; Domain{self.N_c}')
        print(f'Total: {self.N_b*8 + self.N_c}')
        return self.N_b*8 + self.N_c

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
