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

        self.data = params["data"]
        self.path2log = params["path2log"]
        self.path2models = params["path2models"]
        self.plot = params["plot_loss"]
        self.save_loss = params["save_loss"]



        #Initialise best model and loss
        self.best_loss=float('inf')  
        #loss history
        self.current_epoch = 0
        self.current_lr = get_lr(self.optimizer)
        self.losses = {"total":[], "pde": [], "bc": [], "ic": []}

    def MSE_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def IC_loss(self, pred, target, xt):
        pred_u = pred[:, 0:1]
        target_u = target[:, 0:1]

        u_t = torch.autograd.grad(pred_u, xt, torch.ones([pred_u.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 1:2]        #neumann condition

        return (self.MSE_loss(pred_u, target_u) + self.MSE_loss(u_t, torch.zeros_like(u_t)))*100.0

    def BC_loss(self, pred, target):
        pred_u = pred[:, 0:1]
        target_u = target[:, 0:1]
        return self.MSE_loss(pred_u, target_u)*100.0
    
    def PDE_loss(self, pred, target):
            xt = target
            u= pred[:, 0:1]
            #time derivative
            u_t = torch.autograd.grad(u, xt, torch.ones([u.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 1:2]
            u_tt = torch.autograd.grad(u_t, xt, torch.ones([u_t.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 1:2]

            #first order derivative
            u_x = torch.autograd.grad(u, xt, torch.ones([u.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 0:1]
            #second order derivative
            u_xx = torch.autograd.grad(u_x, xt, torch.ones([u_x.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 0:1]

            #PDE loss
            f0 = u_tt - 4*u_xx
            mse_f0 = self.MSE_loss(f0, torch.zeros_like(f0))
            return mse_f0


    def closure(self):
        total_batch_loss = 0.0
        col_batch_loss = 0.0
        bc_batch_loss = 0.0
        ic_batch_loss = 0.0

        self.model.train()

        [col_data, bc_data, ic_data] = self.data[0], self.data[1], self.data[2]
        for i, (col_data, bc_data, ic_data) in enumerate(zip_longest(col_data, bc_data, ic_data)):
            total_loss = torch.tensor(0.0).to(device)
            self.optimizer.zero_grad()
            #unpack data
            col_xt = col_data
            bc_xt, bc_u = bc_data[0], bc_data[1]
            ic_xt, ic_u = ic_data[0], ic_data[1]

            #forward pass
            if col_xt is not None:
                col_xt = col_xt.clone()
                col_xt.requires_grad = True
                col_pred = self.model(col_xt)
                col_loss = self.PDE_loss(col_pred, col_xt)
                total_loss += col_loss
            if bc_xt is not None:
                bc_pred = self.model(bc_xt)
                bc_loss = self.BC_loss(bc_pred, bc_u)
                total_loss += bc_loss
            if ic_xt is not None:
                ic_xt = ic_xt.clone()
                ic_xt.requires_grad = True
                ic_pred = self.model(ic_xt)
                ic_loss = self.IC_loss(ic_pred, ic_u, ic_xt)
                total_loss += ic_loss


            total_loss.backward()
            if self.optimizer.__class__.__name__ != "LBFGS":
                self.optimizer.step()
                
            
            total_batch_loss += total_loss.detach().cpu()
            col_batch_loss += col_loss.detach().cpu()
            bc_batch_loss += bc_loss.detach().cpu()
            ic_batch_loss += ic_loss.detach().cpu()
            



        self.losses["total"].append(total_batch_loss)
        self.losses["pde"].append(col_batch_loss)
        self.losses["bc"].append(bc_batch_loss)
        self.losses["ic"].append(ic_batch_loss)

        if total_batch_loss < self.best_loss:
            self.best_loss = total_batch_loss
            torch.save(self.model.state_dict(), self.path2models + "best_weight.pt")
        if self.save_epochs is not None and self.current_epoch%self.save_epochs == 0:
            torch.save(self.model.state_dict(), self.path2models + "epoch_{}.pt".format(self.current_epoch))
        if self.plot is not None and self.current_epoch%self.plot == 0:
            plotLoss(self.losses, self.path2log + "loss_curve.png", ["Total", "PDE", "BC", "IC"])
        if self.save_loss:
            result_file = open(self.path2log + "running_result.txt","a")
            result_file.write('Epoch {}/{}, current lr={}'.format(self.current_epoch, self.num_epochs - 1, self.current_lr))
            result_file.write('\n')
            result_file.write("Total: %.6f, PDE: %.6f %%, BC: %.6f %%, IC: %.6f %%" %(total_batch_loss, col_batch_loss, bc_batch_loss, ic_batch_loss))
            result_file.write('\n')
            result_file.write("-"*10)
            result_file.write('\n')
            result_file.close()

            #save history to pickle
            loss_file = open(self.path2log + "loss.pkl", "wb")
            pickle.dump(self.losses, loss_file)
            loss_file.close()


        print(
            f"\r{self.current_epoch} Loss: {total_batch_loss:.5e} pde: {col_batch_loss:.3e} bc: {bc_batch_loss:.3e} ic: {ic_batch_loss:.3e}",
            end="",
        )
        if self.current_epoch % 500 == 0:
            print("")



        self.current_epoch += 1
        return total_batch_loss



