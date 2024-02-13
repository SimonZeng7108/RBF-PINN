import os
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import torch
from torch.utils.data import DataLoader
from torch.autograd import grad
from Utils import plotLoss, plot_XYZ_2D
from dataset import Geo_Dataset, Col_Data, BC_Data
from network import DNN_custom, RBF_DNN
from trainer import Trainer
from datetime import date, datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
#Trainig HyperParameters
Batch_Learning = False
Fixed_Seed = True
Save_loss = True
epochs = 20000

Plot_loss = 1000    #plot loss every plot_epoch
save_epochs = 2000 # save model every save_epochs
file_name = 'Lorenz_DNN_100X5'
path2logs = "./logs/"
path2models = "./logs/models/{}_{}/".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), file_name) 




#Model HyperParameters
dim_in = 1 #input shape; e.g. t
dim_out = 3 #output shape; e.g. x , y, z
hidden_layer_centres = [128, 50, 50, 50, 50] # if customised layer 
Pol_Feat = 0 #Numbe of polynomial features



if Batch_Learning:
    Batchsize = 500
else:
    Batchsize = None

if Fixed_Seed == True:
    torch.manual_seed(1234)
    np.random.seed(1234)
else:
    pass




#Initialise DataSet
dataset = Geo_Dataset()
ic_t, ic_xyz, sample_t, sample_xyz, col_t = dataset()


#Initialise DataLoader
col_loader = DataLoader(Col_Data(col_t), batch_size=[len(col_t) if not Batch_Learning else Batchsize][0], shuffle=False)
sample_loader = DataLoader(BC_Data(sample_t, sample_xyz), batch_size=[len(sample_t) if not Batch_Learning else Batchsize][0], shuffle=False)
ic_loader = DataLoader(BC_Data(ic_t, ic_xyz), batch_size=[len(ic_t) if not Batch_Learning else Batchsize][0], shuffle=False)



#Initialise Model
# model = RBF(dim_in=dim_in, dim_out=dim_out, n_layer=n_layer, n_node=n_node, ub=dataset.ub, lb=dataset.lb).to(device).to(torch.float64)
model = RBF_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layer_centres=hidden_layer_centres, ub=dataset.ub, lb=dataset.lb, order_pol=Pol_Feat).to(device).to(torch.float64)
alpha = torch.tensor([1.0], requires_grad=True).to(torch.float64).to(device)
beta = torch.tensor([1.0], requires_grad=True).to(torch.float64).to(device)
rho = torch.tensor([1.0], requires_grad=True).to(torch.float64).to(device)
alpha = torch.nn.Parameter(alpha)   #True value is 10
beta = torch.nn.Parameter(beta)     #True value is 8/3
rho = torch.nn.Parameter(rho)       #True value is 28
model.register_parameter("alpha", alpha)
model.register_parameter("beta", beta)
model.register_parameter("rho", rho)


#Initialise Optimizer
adam = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_sheduler = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=0.9)

if not Batch_Learning:
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=20000,max_eval=None,tolerance_grad=1e-8,tolerance_change=0,history_size=100)

#Initalise Trainer
params_train={
    "num_epochs": epochs,
    "save_epochs": save_epochs,
    "model": model,
    "optimizer": adam,
    'lr_scheduler': lr_sheduler,
    "data": [col_loader, sample_loader, ic_loader],
    "path2log": path2logs,
    "path2models": path2models,
    "plot_loss": Plot_loss,
    "save_loss": Save_loss,
    "params": {"alpha": alpha, "beta": beta, "rho": rho},
}

if __name__ == "__main__":
    if not os.path.exists(path2logs):
        os.mkdir(path2logs)
    if not os.path.exists(path2models):
        os.mkdir(path2models)

    print("---Start Training---")
    start = timer()
    print("---Initialising Trainer---")
    trainer = Trainer(params_train)

    print("---Training with adam---")
    for epoch in range(epochs):
        trainer.closure()

    if not Batch_Learning:
        print("---Training with lbfgs---")
        trainer.optimizer = lbfgs
        trainer.optimizer.step(trainer.closure)

    end = timer()
    time_elapsed = end - start
    print("---Training Finished---")
    print("Training time: ", timedelta(seconds = time_elapsed))
