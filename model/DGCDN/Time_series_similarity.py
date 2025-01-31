import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class time_delay_self_attn(nn.Module):
    def __init__(self, seq_len, channels, heads=1, activation=None, abs=False, device='cpu'):
        super(time_delay_self_attn,self).__init__()
        self.heads = heads
        omega = [torch.eye(seq_len*channels).unsqueeze(0) for _ in range(heads)] # for Time delay attn
        self.omega = nn.Parameter(torch.cat(omega, dim=0).to(device))
        self.activation = activation
        self.abs = abs

    def forward(self,x):
        # roll cosine similarity
        B,C,N,S = x.size()
        ori_x = x.transpose(1,2).reshape(B,N,-1).unsqueeze(1).repeat(1,self.heads,1,1)
        roll_x = [torch.roll(x,shifts=i,dims=3).transpose(1,2).reshape(B,N,-1).unsqueeze(1) for i in range(self.heads)]
        roll_x = torch.cat(roll_x,dim=1)
        # upper
        upper = torch.matmul(ori_x,self.omega)
        upper = torch.matmul(upper,roll_x.transpose(-1,-2))
        # lower
        x_s1 = (ori_x * ori_x).sum(-1)
        x_s1 = x_s1.unsqueeze(2).repeat(1,1,N,1)
        x_s1 = torch.sqrt(x_s1) # relu
        x_s2 = (roll_x * roll_x).sum(-1)
        x_s2 = x_s2.unsqueeze(3).repeat(1,1,1,N)
        x_s2 = torch.sqrt(x_s2) # relu
        lower = x_s1 * x_s2
        # normalized corr
        corrs = upper/lower
        if self.activation is not None:
            corrs = self.activation(corrs)
        if self.abs:
            corrs = corrs.abs()
        return corrs

class dist_measurements(nn.Module):
    def __init__(self, seq_len, channels, heads=1, device='cpu'):
        super(dist_measurements,self).__init__()
        self.heads = heads
        n_auto = int(heads/2) # for distance measurements
        len_prei = int(np.ceil((seq_len+1)/2))
        Omega_auto = [torch.eye(seq_len * channels).unsqueeze(0) for _ in range(n_auto)]
        self.Omega_auto = nn.Parameter(torch.cat(Omega_auto, dim=0).to(device))
        Omega_preiod = [torch.eye(len_prei * channels).unsqueeze(0) for _ in range(heads-n_auto)]
        self.Omega_preiod = nn.Parameter(torch.cat(Omega_preiod, dim=0).to(device))
    
    def forward(self,x):
        dist = []
        for i in range(self.heads):
            if i%2 == 0:
                dist.append(period_distance(x,self.Omega_preiod[i//2]))
            else:
                dist.append(auto_corr_distance(x,self.Omega_auto[i//2]))
        dist_simi = 1 - torch.cat(dist,dim=1)
        return dist_simi
    
def period_distance(x,Omega):
    # sequence period distance
    x_fft = torch.fft.rfft(x, dim = -1)
    x_fft_abs = torch.abs(x_fft)
    period_dist = Euclidean_distance(x_fft_abs,Omega)
    return period_dist.unsqueeze(1)

def auto_corr_distance(x,Omega):
    # sequence auto-correlation distance
    x_fft = torch.fft.rfft(x, dim = -1)
    res = x_fft * torch.conj(x_fft)
    auto_corr = torch.fft.irfft(res, dim = -1)
    auto_corr_dist = Euclidean_distance(auto_corr,Omega)
    return auto_corr_dist.unsqueeze(1)

def Euclidean_distance(x,Omega):
    # Euclidean distance with max-min regularity
    B,C,N,S = x.size()
    x = x.transpose(1,2).reshape(B,N,-1)
    rxry = torch.matmul(x,Omega)
    rxry = torch.matmul(rxry,x.transpose(-1,-2))
    rx_square = torch.matmul(x,Omega)
    rx_square = (rx_square * rx_square).sum(-1)
    rx_square_1 = rx_square.unsqueeze(1).repeat(1,N,1)
    rx_square_2 = rx_square.unsqueeze(2).repeat(1,1,N)
    dist = (rx_square_1 + rx_square_2 - 2*rxry)
    dist = torch.sqrt(F.relu(dist))
    #　使用最大值归一化, 是为了避免 sigmoid(0) = 0.5的问题.
    max_v,_ = dist.reshape(B,-1).max(1)
    max_v = torch.where(max_v==0.0,torch.ones_like(max_v),max_v)
    dist = dist / max_v.unsqueeze(-1).unsqueeze(-1)
    return dist