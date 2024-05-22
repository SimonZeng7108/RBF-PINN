import torch
import torch.nn as nn
import numpy as np
import RBF_kernels as rbf
from pyDOE import lhs

torch.backends.cuda.matmul.allow_tf32 = (
    False  # This is for Nvidia Ampere GPU Architechture
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

class Linear_Layer(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x

class DNN_custom(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_layer_centres=[50, 40, 30, 20], ub=0, lb=0, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(Linear_Layer(dim_in, hidden_layer_centres[0], activation))
        for i in range(len(hidden_layer_centres)-1):
            self.net.append(Linear_Layer(hidden_layer_centres[i], hidden_layer_centres[i+1], activation))
        self.net.append(Linear_Layer(hidden_layer_centres[-1], dim_out, activation=None))
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.net.apply(weights_init)  # xavier initialization

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)
        out = x
        for layer in self.net:
            out = layer(out)
        return out

# RBF Layer
class RBF_Layer(nn.Module):
    """
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self,ub, lb, in_features, out_features, basis_func,  order_pol, fixed_centroid = False, init_type = 'LHS'):
        super(RBF_Layer, self).__init__()
        self.ub = ub
        self.lb = lb
        self.in_features = in_features
        self.out_features = out_features
        if fixed_centroid == False:
            self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
            init_type = None
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.order = order_pol
        self.reset_parameters(init_type, out_features, in_features)

    def reset_parameters(self, init_type, num_centroid, dim_in):
        if init_type != None:
            if init_type == "Gaussian":
                self.centres = np.random.normal(0, 1, size=(num_centroid, dim_in))
            if init_type == "Uniform":
                self.centres = np.random.uniform(self.lb, self.ub, size=(num_centroid, dim_in))
            if init_type == "LHS":
                
                self.centres = self.lb + (self.ub - self.lb) * lhs(dim_in, num_centroid)
            if init_type == "Linear":
                x = np.linspace( self.x_min,  self.x_max, int(num_centroid**0.5)).reshape(-1, 1)
                y = np.linspace(self.y_min, self.y_max, int(num_centroid**0.5)).reshape(-1, 1)
                x, y = np.meshgrid(x, y)
                x, y = x.reshape(-1, 1), y.reshape(-1, 1)
                self.centres = np.concatenate([x, y], axis=1)
            self.centres = torch.tensor(self.centres, dtype=torch.float64).to(device)
        else:
            nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def poly_features(self, x, order):
        poly_feature = torch.ones(x.shape[0], 1).to(device)
        for i in range(1, order+1):
            poly_feature = torch.cat([poly_feature, (x**i).sum(-1).reshape(-1, 1)], dim=1)
        return poly_feature

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        rbf_feature = self.basis_func(distances)
        all_feature = rbf_feature
        if self.order != 0:
            poly_input = input
            poly_feature = self.poly_features(poly_input, self.order)
            all_feature = torch.cat([rbf_feature, poly_feature], dim=1)
        return all_feature


class RBF_DNN(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_layer_centres=[50, 40, 30, 20], ub=0, lb=0, basis_func= rbf.gaussian, activation=nn.Tanh()):
        super().__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        #input layer
        self.rbf_layers.append(RBF_Layer(ub, lb, dim_in, hidden_layer_centres[0], basis_func))
        #hidden layers
        for i in range(len(hidden_layer_centres)-1):
            self.linear_layers.append(Linear_Layer(hidden_layer_centres[i], hidden_layer_centres[i+1], activation))
        self.linear_layers.append(Linear_Layer(hidden_layer_centres[-1], dim_out, activation=None))
        self.rbf_layers.apply(weights_init)  # xavier initialization
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.norm_rbf = nn.BatchNorm1d(hidden_layer_centres[1], affine=False) #affine = False means no learnable parameters


    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)
        out = x
        out = self.rbf_layers[0](out)
        for i in range(len(self.linear_layers)):
            out = self.linear_layers[i](out)
            if i == 0:
                out = self.norm_rbf(out)
        return out
    




if __name__ == '__main__':
    x_min = 0
    x_max = 2
    t_min = 0
    t_max = 5
    ub = np.array([x_max, t_max])
    lb = np.array([x_min, t_min])
         

    
    # Collocation points
    xt_f = np.random.uniform(lb, ub, (1000, 2))
    xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)

    # net = DNN(dim_in=2, dim_out=1, n_layer=5, n_node=40, ub=ub, lb=lb).to(device)
    # u = net(xt_f)
    # print(u.shape)

    # net = RBF(dim_in=2, dim_out=1, n_layer=5, n_node=40, ub=ub, lb=lb).to(device)
    # u = net(xt_f)
    # print(u.shape)

    net = RBF_DNN(2, 1, hidden_layer_centres=[50, 50, 50, 50], ub=ub, lb=lb, basis_func= rbf.gaussian, activation=nn.Tanh()).to(device)
    u = net(xt_f)


# import torch
# import torch.nn as nn
# import numpy as np
# import RBF_kernels as rbf

# torch.backends.cuda.matmul.allow_tf32 = (
#     False  # This is for Nvidia Ampere GPU Architechture
# )
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight.data)
#         torch.nn.init.zeros_(m.bias.data)
        



# class Linear_Layer(nn.Module):
#     def __init__(self, n_in, n_out, activation):
#         super().__init__()
#         self.layer = nn.Linear(n_in, n_out)
#         self.activation = activation

#     def forward(self, x):
#         x = self.layer(x)
#         if self.activation:
#             x = self.activation(x)
#         return x

# class DNN(nn.Module):
#     def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
#         super().__init__()
#         self.net = nn.ModuleList()
#         self.net.append(Linear_Layer(dim_in, n_node, activation))
#         for _ in range(n_layer):
#             self.net.append(Linear_Layer(n_node, n_node, activation))
#         self.net.append(Linear_Layer(n_node, dim_out, activation=None))
#         self.ub = torch.tensor(ub, dtype=torch.float).to(device)
#         self.lb = torch.tensor(lb, dtype=torch.float).to(device)
#         self.net.apply(weights_init)  # xavier initialization

#     def forward(self, x):
#         x = (x - self.lb) / (self.ub - self.lb)  # Min-max scaling
#         out = x
#         for layer in self.net:
#             out = layer(out)
#         return out

# class DNN_custom(nn.Module):
#     def __init__(self, dim_in, dim_out, hidden_layer_centres=[50, 40, 30, 20], ub=0, lb=0, activation=nn.Tanh()):
#         super().__init__()
#         self.net = nn.ModuleList()
#         self.net.append(Linear_Layer(dim_in, hidden_layer_centres[0], activation))
#         for i in range(len(hidden_layer_centres)-1):
#             self.net.append(Linear_Layer(hidden_layer_centres[i], hidden_layer_centres[i+1], activation))
#         self.net.append(Linear_Layer(hidden_layer_centres[-1], dim_out, activation=None))
#         self.ub = torch.tensor(ub, dtype=torch.float).to(device)
#         self.lb = torch.tensor(lb, dtype=torch.float).to(device)
#         self.net.apply(weights_init)  # xavier initialization

#     def forward(self, x):
#         x = (x - self.lb) / (self.ub - self.lb)
#         out = x
#         for layer in self.net:
#             out = layer(out)
#         return out

# class MahalanobisDistance(nn.Module):
#     def __init__(self):
#         super(MahalanobisDistance, self).__init__()
#         self.mu = None
#         self.cov = None
#         self.inverse_cov = None

#     def forward(self, x, c):
#         """
#         Calculates the squared Mahalanobis distance between x and centroid
#         """

#         import time

#         x, c = torch.Tensor(x), torch.Tensor(c)
#         size = (x.shape[0], c.shape[0], x.shape[1])
#         x = x.clone().unsqueeze(1).expand(size)
#         self.mu = c.clone().unsqueeze(0).expand(size)

#         self.cov = self.get_cov(x, self.mu)
#         self.inverse_cov = self.get_inverse_cov(self.cov)
#         delta = x -  self.mu
#         delta = torch.swapaxes(delta, 0, 1)

#         delta_inv_cov = torch.bmm(delta, self.inverse_cov)
        
#         start = time.time()
#         delta_inv_cov_deltaT = torch.bmm(delta_inv_cov, delta.permute(0, 2, 1))
#         end = time.time()
#         print("time: ", end - start)


#         mahalanobis_distance = torch.sqrt(torch.diagonal(delta_inv_cov_deltaT, dim1 = 1, dim2 = 2))

#         return mahalanobis_distance.permute(1, 0)
    
#     def get_cov(self, x, mu):
#         """
#         Calculates covariance matrix 
#         (1/(n - 1)) * Sum((X-mu)^T * (x - mu))

#         Reference 
#         ---------
#         - https://en.wikipedia.org/wiki/Covariance
#         """

#         n = x.size(0)
#         delta = x - mu

#         delta = torch.swapaxes(delta, 0, 1)
#         cov = (1/(n-1)) * delta.permute(0, 2, 1).bmm(delta)

#         return cov 

#     def get_inverse_cov(self, cov):
#         inverse_cov = torch.pinverse(cov)
#         return inverse_cov

# # RBF Layer
# class RBF_Layer(nn.Module):
#     """
#     Transforms incoming data using a given radial basis function:
#     u_{i} = rbf(||x - c_{i}|| / s_{i})

#     Arguments:
#         in_features: size of each input sample
#         out_features: size of each output sample

#     Shape:
#         - Input: (N, in_features) where N is an arbitrary batch size
#         - Output: (N, out_features) where N is an arbitrary batch size

#     Attributes:
#         centres: the learnable centres of shape (out_features, in_features).
#             The values are initialised from a standard normal distribution.
#             Normalising inputs to have mean 0 and standard deviation 1 is
#             recommended.
        
#         log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
        
#         basis_func: the radial basis function used to transform the scaled
#             distances.
#     """

#     def __init__(self, in_features, out_features, basis_func):
#         super(RBF_Layer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
#         self.basis_func = basis_func
#         self.mahalanobis_d = MahalanobisDistance()
#         self.reset_parameters()


#     def reset_parameters(self):
#         nn.init.normal_(self.centres, 0, 1)
#         nn.init.constant_(self.log_sigmas, 0)


#     def forward(self, input):
#         x = input
#         c = self.centres

#         maha_distance = self.mahalanobis_d(x, c)
#         distances = maha_distance / torch.exp(self.log_sigmas).unsqueeze(0)
    
#         return self.basis_func(distances)

# class RBF(nn.Module):
#     def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, Linear_embed =True, basis_func= rbf.gaussian,activation=nn.Tanh()):
#         super(RBF, self).__init__()
#         self.le = Linear_embed
#         self.rbf_layers = nn.ModuleList()
#         #input layer
#         self.rbf_layers.append(RBF_Layer(dim_in, n_node, basis_func))
#         #hidden layers
#         for _ in range(n_layer):
#             self.rbf_layers.append(RBF_Layer(n_node, n_node, basis_func))
#         #output layer
#         if Linear_embed:
#             self.rbf_layers.append(RBF_Layer(n_node, n_node, basis_func))
#         else:
#             self.rbf_layers.append(RBF_Layer(n_node, dim_out, basis_func))


#         if Linear_embed:
#             self.linear_layers = nn.ModuleList()
#             self.linear_layers.append(Linear_Layer(n_node, n_node,activation)) #input
#             for _ in range(n_layer):                                           #hidden
#                 self.linear_layers.append(Linear_Layer(n_node, n_node, activation))
#             self.linear_layers.append(Linear_Layer(n_node, dim_out, activation=None)) #output
#             self.linear_layers.apply(weights_init)  # xavier initialization

#         self.ub = torch.tensor(ub, dtype=torch.float).to(device)
#         self.lb = torch.tensor(lb, dtype=torch.float).to(device)
    
#     def forward(self, x):
#         x = (x - self.lb) / (self.ub - self.lb)  # Min-max scaling
#         out = x
#         for i in range(len(self.rbf_layers)):
#             out = self.rbf_layers[i](out)
#             if self.le:
#                 out = self.linear_layers[i](out)
#         return out
    

# class RBF_DNN(nn.Module):
#     def __init__(self, dim_in, dim_out, hidden_layer_centres=[50, 40, 30, 20], ub=0, lb=0, basis_func= rbf.gaussian, activation=nn.Tanh()):
#         super().__init__()
#         self.rbf_layers = nn.ModuleList()
#         self.linear_layers = nn.ModuleList()
#         #input layer
#         self.rbf_layers.append(RBF_Layer(dim_in, hidden_layer_centres[0], basis_func))
#         #hidden layers
#         for i in range(len(hidden_layer_centres)-1):
#             self.linear_layers.append(Linear_Layer(hidden_layer_centres[i], hidden_layer_centres[i+1], activation))
#         self.linear_layers.append(Linear_Layer(hidden_layer_centres[-1], dim_out, activation=None))
#         self.rbf_layers.apply(weights_init)  # xavier initialization
#         self.ub = torch.tensor(ub, dtype=torch.float).to(device)
#         self.lb = torch.tensor(lb, dtype=torch.float).to(device)

#     def forward(self, x):
#         x = (x - self.lb) / (self.ub - self.lb)
#         out = x
#         out = self.rbf_layers[0](out)

#         for i in range(len(self.linear_layers)):
#             out = self.linear_layers[i](out)
#         return out
    




# if __name__ == '__main__':
#     x_min = 0
#     x_max = 2
#     t_min = 0
#     t_max = 5
#     ub = np.array([x_max, t_max])
#     lb = np.array([x_min, t_min])

#     # Collocation points
#     xt_f = np.random.uniform(lb, ub, (1000, 2))
#     xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)

#     # net = DNN(dim_in=2, dim_out=1, n_layer=5, n_node=40, ub=ub, lb=lb).to(device)
#     # u = net(xt_f)
#     # print(u.shape)

#     net = RBF_DNN(2, 1, hidden_layer_centres=[50, 50, 50, 50], ub=ub, lb=lb, basis_func= rbf.gaussian, activation=nn.Tanh()).to(device)
#     u = net(xt_f)
#     print(u.shape)
