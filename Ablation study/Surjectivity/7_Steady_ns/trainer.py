import torch
from torch.nn import functional as F
import numpy as np
import copy
import pickle
from itertools import zip_longest
from Utils import plotLoss, get_lr
from network import DNN



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
        self.Re = params["physics"]["Re"]


        #Initialise best model and loss
        self.best_loss=float('inf')  
        #loss history
        self.current_epoch = 0
        self.current_lr = get_lr(self.optimizer)
        self.losses = {"total":[], "pde": [], "bc": [], "outlet": []}

    def MSE_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def Outlet_loss(self, pred, target):
        pred_p = pred[:, 2:3]
        target_p = target[:, 0:1]
        return self.MSE_loss(pred_p, target_p)

    def BC_loss(self, pred, target):
        pred_u = pred[:, 0:1]
        pred_v = pred[:, 1:2]
        target_u = target[:, 0:1]
        target_v = target[:, 1:2]
        return self.MSE_loss(pred_u, target_u) + self.MSE_loss(pred_v, target_v)
    
    def PDE_loss(self, pred, target):
            xy = target
            u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
            #first order derivative
            u_x = torch.autograd.grad(u, xy, torch.ones([u.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 0:1]
            u_y = torch.autograd.grad(u, xy, torch.ones([u.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 1:2]
            v_x = torch.autograd.grad(v, xy, torch.ones([v.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 0:1]
            v_y = torch.autograd.grad(v, xy, torch.ones([v.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 1:2]
            p_x = torch.autograd.grad(p, xy, torch.ones([p.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 0:1]
            p_y = torch.autograd.grad(p, xy, torch.ones([p.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 1:2]
            #second order derivative
            u_xx = torch.autograd.grad(u_x, xy, torch.ones([u_x.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 0:1]
            u_yy = torch.autograd.grad(u_y, xy, torch.ones([u_y.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 1:2]
            v_xx = torch.autograd.grad(v_x, xy, torch.ones([v_x.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 0:1]
            v_yy = torch.autograd.grad(v_y, xy, torch.ones([v_y.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0][:, 1:2]
            #continous equation
            f0 = u_x + v_y
            #momentum equation
            f1 = u*u_x + v*u_y + p_x - 1/self.Re*(u_xx + u_yy)
            f2 = u*v_x + v*v_y + p_y - 1/self.Re*(v_xx + v_yy)

            mse_f0 = self.MSE_loss(f0, torch.zeros_like(f0))
            mse_f1 = self.MSE_loss(f1, torch.zeros_like(f1))
            mse_f2 = self.MSE_loss(f2, torch.zeros_like(f2))

            return mse_f0 + mse_f1 + mse_f2


    def closure(self):
        total_batch_loss = 0.0
        col_batch_loss = 0.0
        bc_batch_loss = 0.0
        outlet_batch_loss = 0.0

        self.model.train()

        [col_data, bc_data, outlet_data] = self.data[0], self.data[1], self.data[2]
        for i, (col_data, bc_data, outlet_data) in enumerate(zip_longest(col_data, bc_data, outlet_data)):
            total_loss = torch.tensor(0.0).to(device)
            self.optimizer.zero_grad()
            #unpack data
            col_xy = col_data
            bc_xy, bc_uv = bc_data[0], bc_data[1]#
            out_xy, out_p = outlet_data[0], outlet_data[1]

            #forward pass
            if col_xy is not None:
                col_xy = col_xy.clone()
                col_xy.requires_grad = True
                col_pred = self.model(col_xy)
                col_loss = self.PDE_loss(col_pred, col_xy)
                total_loss += col_loss
            if bc_xy is not None:
                bc_pred = self.model(bc_xy)
                bc_loss = self.BC_loss(bc_pred, bc_uv)
                total_loss += bc_loss
            if out_xy is not None:
                out_pred = self.model(out_xy)
                outlet_loss = self.Outlet_loss(out_pred, out_p)
                total_loss += outlet_loss



            total_loss.backward()
            if self.optimizer.__class__.__name__ != "LBFGS":
                self.optimizer.step()
                
            
            total_batch_loss += total_loss.detach().cpu()
            col_batch_loss += col_loss.detach().cpu()
            bc_batch_loss += bc_loss.detach().cpu()
            outlet_batch_loss += outlet_loss.detach().cpu()

            



        self.losses["total"].append(total_batch_loss)
        self.losses["pde"].append(col_batch_loss)
        self.losses["bc"].append(bc_batch_loss)
        self.losses["outlet"].append(outlet_batch_loss)


        if total_batch_loss < self.best_loss:
            self.best_loss = total_batch_loss
            torch.save(self.model.state_dict(), self.path2models + "best_weight.pt")
        if self.save_epochs is not None and self.current_epoch%self.save_epochs == 0:
            torch.save(self.model.state_dict(), self.path2models + "epoch_{}.pt".format(self.current_epoch))
        if self.plot is not None and self.current_epoch%self.plot == 0:
            plotLoss(self.losses, self.path2log + "loss_curve.png", ["Total", "PDE", "BC", "Outlet"])
        if self.save_loss:
            result_file = open(self.path2log + "running_result.txt","a")
            result_file.write('Epoch {}/{}, current lr={}'.format(self.current_epoch, self.num_epochs - 1, self.current_lr))
            result_file.write('\n')
            result_file.write("Total: %.6f, PDE: %.6f %%, BC: %.6f %%, Outlet: %.6f %%" %(total_batch_loss, col_batch_loss, bc_batch_loss, outlet_batch_loss))
            result_file.write('\n')
            result_file.write("-"*10)
            result_file.write('\n')
            result_file.close()

            #save history to pickle
            loss_file = open(self.path2log + "loss.pkl", "wb")
            pickle.dump(self.losses, loss_file)
            loss_file.close()


        print(
            f"\r{self.current_epoch} Loss: {total_batch_loss:.5e} pde: {col_batch_loss:.3e} bc: {bc_batch_loss:.3e}, outlet: {outlet_batch_loss:.3e}",
            end="",
        )
        if self.current_epoch % 500 == 0:
            print("")



        self.current_epoch += 1
        return total_batch_loss



