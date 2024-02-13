import torch
from torch.nn import functional as F
import numpy as np
import copy
import pickle
from itertools import zip_longest
from Utils import plotLoss, get_lr



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Trainer:
    def __init__(self, params):
        self.num_epochs = params["num_epochs"]
        self.save_epochs = params["save_epochs"]
        self.model = params["model"]
        self.optimizer = params["optimizer"]
        self.scheduler = params["lr_scheduler"]
        self.data = params["data"]
        self.path2log = params["path2log"]
        self.path2models = params["path2models"]
        self.plot = params["plot_loss"]
        self.save_loss = params["save_loss"]
        self.mu1 = params["params"]["mu1"]
        self.mu2 = params["params"]["mu2"]


        #Initialise best model and loss
        self.best_loss=float('inf')  
        #loss history
        self.current_epoch = 0
        self.current_lr = get_lr(self.optimizer)
        self.losses = {"total":[], "pde": []}

    def MSE_loss(self, pred, target):
        return F.mse_loss(pred, target)
    
    

    def PDE_loss(self, pred, col_xt):
            xt = col_xt
            u= pred[:, 0:1]
            
            #first order derivative 
            u_x = torch.autograd.grad(u, xt, torch.ones([u.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 0:1]

            #time derivative
            u_t = torch.autograd.grad(u, xt, torch.ones([u.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 1:2]

            #second order derivative
            u_xx = torch.autograd.grad(u_x, xt, torch.ones([u_x.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 0:1]

            #PDE loss
            f0 = u_t + self.mu1*u*u_x - self.mu2*u_xx
            mse_f0 = self.MSE_loss(f0, torch.zeros_like(f0))
            return  mse_f0



    def closure(self):
        total_batch_loss = 0.0
        col_batch_loss = 0.0

        self.model.train()

        col_data = self.data[0]

        for i, col_data in enumerate(col_data):
            total_loss = torch.tensor(0.0).to(device)
            self.optimizer.zero_grad()
            #unpack data
            col_xt, col_u = col_data[0], col_data[1]
            
            #forward pass
            col_xt.requires_grad = True
            col_pred = self.model(col_xt)
            col_loss = self.PDE_loss(col_pred, col_xt)
            data_loss = self.MSE_loss(col_pred, col_u)
            all_loss = col_loss + data_loss
            total_loss += all_loss


            total_loss.backward()
            if self.optimizer.__class__.__name__ != "LBFGS":
                self.optimizer.step()
                
            
            total_batch_loss += total_loss.detach().cpu()
            col_batch_loss += col_loss.detach().cpu()


        # if (self.current_epoch+1) % 1000 == 0:
        #     self.scheduler.step()

        self.losses["total"].append(total_batch_loss)
        self.losses["pde"].append(col_batch_loss)

        if total_batch_loss < self.best_loss:
            self.best_loss = total_batch_loss
            torch.save(self.model.state_dict(), self.path2models + "best_weight.pt")
        if self.save_epochs is not None and self.current_epoch%self.save_epochs == 0:
            torch.save(self.model.state_dict(), self.path2models + "epoch_{}.pt".format(self.current_epoch))
        if self.plot is not None and self.current_epoch%self.plot == 0:
            plotLoss(self.losses, self.path2log + "loss_curve.png", ["Total", "PDE"])
        # if self.save_loss:
        #     result_file = open(self.path2log + "running_result.txt","a")
        #     result_file.write('Epoch {}/{}, current lr={}'.format(self.current_epoch, self.num_epochs - 1, self.current_lr))
        #     result_file.write('\n')
        #     result_file.write("Total: %.6f, PDE: %.6f %%, sample: %.6f %%, IC: %.6f %%" %(total_batch_loss, col_batch_loss, sample_batch_loss, ic_batch_loss))
        #     result_file.write('\n')
        #     result_file.write("-"*10)
        #     result_file.write('\n')
        #     result_file.close()

            # #save history to pickle
            # loss_file = open(self.path2log + "loss.pkl", "wb")
            # pickle.dump(self.losses, loss_file)
            # loss_file.close()


        print(
            f"\r{self.current_epoch} Loss: {total_batch_loss:.5e} pde: {col_batch_loss:.3e}",
            end="",
        )
        if self.current_epoch % 500 == 0:
            print("")
            print("mu1: ", self.mu1.detach().cpu().item())
            print("mu2: ", self.mu2.detach().cpu().item())

        self.current_epoch += 1
        return total_batch_loss



