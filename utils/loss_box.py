import torch
import numpy as np
import torch.nn as nn

class Graph_Contrastive_Loss(nn.Module):
    def __init__(self, cl_sparity_rate, disturb_indices_bool=True, 
                 indices_disturb_type='samble_only', limite_min_value_bool=True, eps=1e-5):
        super(Graph_Contrastive_Loss,self).__init__()   
        # module bool
        self.disturb_indices_bool = disturb_indices_bool
        self.indices_disturb_type = indices_disturb_type
        self.limite_min_value_bool = limite_min_value_bool
        # hyper params
        self.alpha = 1 - cl_sparity_rate
        self.eps = eps
        self.cl_sparity_rate = cl_sparity_rate
    
    def disturb_indices(self,mat,training):
        if self.disturb_indices_bool and training:
            if self.indices_disturb_type == 'all':
                batch_shuffle = torch.randperm(mat.size(0))
                sample_shuffle = torch.randperm(mat.size(1))
                mat = mat[batch_shuffle,sample_shuffle,:,:]
            elif self.indices_disturb_type == 'samble_only':
                sample_shuffle = torch.randperm(mat.size(1))
                mat = mat[:,sample_shuffle,:,:]
        return mat
    
    def mins_values_limite(self,mat1,mat2):
        if self.limite_min_value_bool:
            mat1 = (1-2*self.eps)*mat1+self.eps
            mat2 = (1-2*self.eps)*mat2+self.eps
        return mat1,mat2
    
    def Balanced_CE_loss(self,x,y,alpha=0.5):
        loss = -(alpha)*(y)*torch.log(x) - (1-alpha)*(1-y)*torch.log(1-x)
        return loss
    
    def CL_loss(self,mat1,mat2):
        B,H,N,_ = mat1.size()
        # label
        diff = mat1-mat2
        mins_values = diff.reshape((B,H,-1)).quantile(q=1-self.cl_sparity_rate,dim=-1).unsqueeze(2).unsqueeze(2)
        mark = (diff < mins_values)
        label = torch.where(mark,torch.zeros_like(diff),torch.ones_like(diff))
        # loss
        loss1 = self.Balanced_CE_loss(mat1, label, self.alpha)
        loss2 = self.Balanced_CE_loss(mat2, 1-label, 1-self.alpha)
        loss = loss1 + loss2
        return loss
    
    def forward(self, mat1, mat2, training):
        mat1 = mat1.abs()
        mat2 = self.disturb_indices(mat2, training)
        mat1,mat2 = self.mins_values_limite(mat1,mat2)
        loss = self.CL_loss(mat1,mat2)
        if torch.isnan(loss.mean()):
            print('hear')
        return loss.mean().reshape(1)

def unite_loss(model_loss, graph_loss, rate=1.0):
    return model_loss + rate * graph_loss

def mse(preds, labels, **args):
    return ((preds-labels)**2).mean()

def mae(preds, labels,**args):
    return torch.abs(preds-labels).mean()

def masked_mse(preds, labels, null_val=np.nan, dim='all'):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if dim == 'all':
        return torch.mean(loss,dim=(1, 2, 3))
    elif dim == 'time_ept':
        return torch.mean(loss,dim=(1, 2))

def masked_rmse(preds, labels, null_val=np.nan, dim='all'):
    mse = torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val, dim=dim))
    return mse.mean(0)

def masked_mae(preds, labels, null_val=np.nan, dim='all'):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # IPython.embed()
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if dim == 'all':
        mae = torch.mean(loss)
    elif dim == 'time_ept':
        mae = torch.mean(loss, dim=(0, 1, 2))
    return mae

def masked_mape(preds, labels, null_val=np.nan, min_threshold=1e-5, dim='all'):
    zeros_filter = (torch.abs(labels)<min_threshold)
    labels = torch.where(zeros_filter, torch.zeros_like(labels), labels)
    zeros_mask = (labels != 0.0)

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask * zeros_mask
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs((preds-labels)/labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    if dim == 'all':
        mape = torch.mean(loss)
    elif dim == 'time_ept':
        mape = torch.mean(loss, dim=(0, 1, 2))
    return mape

def metric(pred, real, null_val=np.nan):
    metrics_dict = {}
    metrics_dict.update({
        'mae':masked_mae(pred,real,null_val).item(),
        'mape' : masked_mape(pred,real,null_val).item(),
        'rmse' : masked_rmse(pred,real,null_val).item(),
        'mse' : masked_mse(pred,real,null_val).mean(0).item(),
        'mae_all' : masked_mae(pred,real,null_val,dim='time_ept').tolist(),
        'mape_all' : masked_mape(pred,real,null_val,dim='time_ept').tolist(),
        'mse_all' : masked_mse(pred,real,null_val,dim='time_ept').mean(0).tolist(),
        'rmse_all' : masked_rmse(pred,real,null_val,dim='time_ept').tolist()
        })
    return metrics_dict