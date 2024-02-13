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
        self.alpha = params["params"]["alpha"]
        self.beta = params["params"]["beta"]
        self.rho = params["params"]["rho"]


        #Initialise best model and loss
        self.best_loss=float('inf')  
        #loss history
        self.current_epoch = 0
        self.current_lr = get_lr(self.optimizer)
        self.losses = {"total":[], "pde": [], "sample": [], "ic": []}

    def MSE_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def IC_loss(self, pred, target):
        pred_xyz = pred
        pred_x, pred_y, pred_z = pred_xyz[:, 0:1], pred_xyz[:, 1:2], pred_xyz[:, 2:3]
        target_xyz = target
        target_x, target_y, target_z = target_xyz[:, 0:1], target_xyz[:, 1:2], target_xyz[:, 2:3]
        return (self.MSE_loss(pred_x, target_x)) + (self.MSE_loss(pred_y, target_y)) + (self.MSE_loss(pred_z, target_z))

    def Sample_loss(self, pred, target):
        pred_xyz = pred
        pred_x, pred_y, pred_z = pred_xyz[:, 0:1], pred_xyz[:, 1:2], pred_xyz[:, 2:3]
        target_xyz = target
        target_x, target_y, target_z = target_xyz[:, 0:1], target_xyz[:, 1:2], target_xyz[:, 2:3]
        return (self.MSE_loss(pred_x, target_x)) + (self.MSE_loss(pred_y, target_y)) + (self.MSE_loss(pred_z, target_z))
    
    def PDE_loss(self, pred, col):
            t = col
            x, y, z = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

            #derivative 
            x_t = torch.autograd.grad(x, t, torch.ones([x.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0]
            y_t = torch.autograd.grad(y, t, torch.ones([y.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0]
            z_t = torch.autograd.grad(z, t, torch.ones([z.shape[0],1]).to(device),retain_graph=True, create_graph=True)[0]

            #lorenz system
            f0 = x_t - self.alpha*(y - x)
            f1 = y_t - x*(self.rho-z) + y
            f2 = z_t - x*y + self.beta*z

            return self.MSE_loss(f0, torch.zeros_like(f0)) + self.MSE_loss(f1, torch.zeros_like(f1)) + self.MSE_loss(f2, torch.zeros_like(f2))


    def closure(self):
        total_batch_loss = 0.0
        col_batch_loss = 0.0
        sample_batch_loss = 0.0
        ic_batch_loss = 0.0

        self.model.train()

        [col_data, sample_data, ic_data] = self.data[0], self.data[1], self.data[2]
        for i, (col_data, sample_data, ic_data) in enumerate(zip_longest(col_data, sample_data, ic_data)):
            total_loss = torch.tensor(0.0).to(device)
            self.optimizer.zero_grad()
            #unpack data
            col_t = col_data
            sample_t, sample_xyz = sample_data[0], sample_data[1]
            ic_t, ic_xyz = ic_data[0], ic_data[1]

            #forward pass
            if col_t is not None:
                col_t = col_t.clone()
                col_t.requires_grad = True
                col_pred = self.model(col_t)
                col_loss = self.PDE_loss(col_pred, col_t)
                total_loss += col_loss
            if sample_t is not None:
                sample_pred = self.model(sample_t)
                sample_loss = self.Sample_loss(sample_pred, sample_xyz)
                total_loss += sample_loss
            if ic_t is not None:
                ic_pred = self.model(ic_t)
                ic_loss = self.IC_loss(ic_pred, ic_xyz)
                total_loss += ic_loss


            total_loss.backward()
            if self.optimizer.__class__.__name__ != "LBFGS":
                self.optimizer.step()
                
            
            total_batch_loss += total_loss.detach().cpu()
            col_batch_loss += col_loss.detach().cpu()
            sample_batch_loss += sample_loss.detach().cpu()
            ic_batch_loss += ic_loss.detach().cpu()
            

        # if (self.current_epoch+1) % 1000 == 0:
        #     self.scheduler.step()

        self.losses["total"].append(total_batch_loss)
        self.losses["pde"].append(col_batch_loss)
        self.losses["sample"].append(sample_batch_loss)
        self.losses["ic"].append(ic_batch_loss)

        # if total_batch_loss < self.best_loss:
        #     self.best_loss = total_batch_loss
        #     torch.save(self.model.state_dict(), self.path2models + "best_weight.pt")
        if self.save_epochs is not None and self.current_epoch%self.save_epochs == 0:
            torch.save(self.model.state_dict(), self.path2models + "epoch_{}.pt".format(self.current_epoch))
        if self.plot is not None and self.current_epoch%self.plot == 0:
            plotLoss(self.losses, self.path2log + "loss_curve.png", ["Total", "PDE", "sample", "IC"])
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
            f"\r{self.current_epoch} Loss: {total_batch_loss:.5e} pde: {col_batch_loss:.3e} sample: {sample_batch_loss:.3e} ic: {ic_batch_loss:.3e}",
            end="",
        )
        if self.current_epoch % 500 == 0:
            print("")
            print("alpha: ", self.alpha.detach().cpu().item())
            print("beta: ", self.beta.detach().cpu().item())
            print("rho: ", self.rho.detach().cpu().item())



        self.current_epoch += 1
        return total_batch_loss



