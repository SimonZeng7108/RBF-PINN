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
file_name = '1d_wave_DNN_100X5'
path2logs = "./logs/"
path2models = "./logs/models/{}_{}/".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), file_name) 


#Model HyperParameters
dim_in = 2 #input shape; e.g. x,t
dim_out = 1 #output shape; e.g. u
hidden_layer_centres = [128, 100, 100, 100, 100, 100] # Feature mapping and Hidden layer shape
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
ic_xt, ic_u, bc_xt, bc_u, col_xt = dataset()


#Initialise DataLoader
col_loader = DataLoader(Col_Data(col_xt), batch_size=[len(col_xt) if not Batch_Learning else Batchsize][0], shuffle=True)
bc_loader = DataLoader(BC_Data(bc_xt, bc_u), batch_size=[len(bc_xt) if not Batch_Learning else Batchsize][0], shuffle=True)
ic_loader = DataLoader(BC_Data(ic_xt, ic_u), batch_size=[len(ic_xt) if not Batch_Learning else Batchsize][0], shuffle=True)



#Initialise Model
model = RBF_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layer_centres=hidden_layer_centres, ub=dataset.ub, lb=dataset.lb, order_pol=Pol_Feat).to(device).to(torch.float64)

#Initialise Optimizer
adam = torch.optim.Adam(model.parameters(), lr=1e-3)
if not Batch_Learning:
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=15000,max_eval=None,tolerance_grad=1e-8,tolerance_change=0,history_size=100)

#Initalise Trainer
params_train={
    "num_epochs": epochs,
    "save_epochs": save_epochs,
    "model": model,
    "optimizer": adam,
    "data": [col_loader, bc_loader, ic_loader],
    "path2log": path2logs,
    "path2models": path2models,
    "plot_loss": Plot_loss,
    "save_loss": Save_loss,
    "physics": {}
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
