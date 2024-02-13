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
file_name = 'Inve_Burgers_DNN_100X5'
path2logs = "./logs/"
path2models = "./logs/models/{}_{}/".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), file_name) 




#Model HyperParameters
dim_in = 2 #input shape; e.g. t
dim_out = 1 #output shape; e.g. x , y, z
hidden_layer_centres = [128, 50, 50, 50, 50] # if customised layer 
Pol_Feat = 10 #Numbe of polynomial features



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
col_xt, col_u = dataset()


#Initialise DataLoader
col_loader = DataLoader(BC_Data(col_xt, col_u), batch_size=[len(col_xt) if not Batch_Learning else Batchsize][0], shuffle=False)



#Initialise Model
# model = RBF(dim_in=dim_in, dim_out=dim_out, n_layer=n_layer, n_node=n_node, ub=dataset.ub, lb=dataset.lb).to(device).to(torch.float64)
model = RBF_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layer_centres=hidden_layer_centres, ub=dataset.ub, lb=dataset.lb, order_pol=Pol_Feat).to(device).to(torch.float64)
mu1 = torch.tensor([1.0], requires_grad=True).to(torch.float64).to(device)
mu2 = torch.tensor([1.0], requires_grad=True).to(torch.float64).to(device)
mu1 = torch.nn.Parameter(mu1)   #True value is 10
mu2 = torch.nn.Parameter(mu2)     #True value is 8/3   #True value is 28
model.register_parameter("mu1", mu1)
model.register_parameter("mu2", mu2)


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
    "data": [col_loader],
    "path2log": path2logs,
    "path2models": path2models,
    "plot_loss": Plot_loss,
    "save_loss": Save_loss,
    "params": {"mu1": mu1, "mu2": mu2},
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
