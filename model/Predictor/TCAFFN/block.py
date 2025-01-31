import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import calculate_order

class fft_mlp(nn.Module):
    def __init__(self,seq_in,seq_out,channels):
        nn.Module.__init__(self)
        self.u_r = Indepent_Linear(seq_in//2 + 1, seq_out, channels)
        self.u_i = Indepent_Linear(seq_in//2 + 1, seq_out, channels)
    def forward(self, x):
        x = torch.fft.rfft(x)
        x = self.u_r(x.real) + self.u_i(x.imag)
        return x

class gated_mlp(nn.Module):
    def __init__(self, seq_in, seq_out, channels, dp_rate=0.3,use_update=True):
        nn.Module.__init__(self)
        self.channels = channels
        self.fft = fft_mlp(seq_in,seq_out,channels)     
        self.update = nn.Linear(seq_out, seq_out)
        self.dropout = nn.Dropout(dp_rate)
        self.use_update = use_update

    def forward(self, x):
        h = self.fft(x)
        if self.use_update:
            h = self.update(x) 
        else:
            h = self.update(h)
        h = F.tanh(h)
        h = self.dropout(h)
        return h
    
class gated_bmlp(nn.Module):
    def __init__(self, seq_in, seq_out, dp_rate=0.3):
        nn.Module.__init__(self)
        self.update = nn.Bilinear(seq_out, seq_out, seq_out)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, x):
        h = self.update(x,x)
        h = F.tanh(h)
        h = self.dropout(h)
        return h

class gated_gcn(nn.Module):
    def __init__(self, seq_len, channels, heads=3, device='cpu', dp_rate=0.3):
        nn.Module.__init__(self)
        self.fc = nn.Linear(heads*seq_len, seq_len)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, x, adjs):
        B,C,N,T = x.size()
        h = torch.einsum('HNM,BMIS->BHNIS', (adjs,x)).permute(0,2,3,1,4).reshape(B,C,N,-1)
        h = F.tanh(h)
        # h = self.dropout(h)
        return h


class TimeEmbedding(nn.Module):
    def __init__(self, s_in, s_out, c_date, channels, time_emb_bool=True, dp_rate=0):
        nn.Module.__init__(self)
        self.time_proj1 = nn.Linear(c_date, channels*3)
        self.time_proj2 = nn.Linear(channels*3, channels)
        self.time_proj3 = nn.Linear(c_date, channels)
        self.mlp_x = Indepent_Linear(s_in, s_in, channels, True)
        self.mlp_y = Indepent_Linear(s_out, s_out, channels, True)
        self.time_emb_bool = time_emb_bool
        self.s_in = s_in
        self.s_out = s_out
        self.channels = channels
        self.dropout = nn.Dropout(dp_rate)

    def share_proj(self, x_mark):
        x_t = self.time_proj1(x_mark)
        x_t = F.relu(x_t)
        x_t = self.time_proj2(x_t).unsqueeze(2).transpose(1,3)
        return x_t
    
    def forward(self, x_mark, y_mark):
        if not self.time_emb_bool:
            B,_,_ = x_mark.size()
            x_t = torch.zeros((B,self.channels,1,self.s_in),device=x_mark.device) #(b,c,n,t_in)
            y_t = torch.zeros((B,self.channels,1,self.s_out),device=x_mark.device) #(b,c,n,t_out)
        else:
            x_t = self.share_proj(x_mark)
            y_t = self.share_proj(y_mark)
            x_t = self.mlp_x(x_t)
            y_t = self.mlp_y(y_t)
        return x_t, y_t


class FreqConv(nn.Module):
    def __init__(self, c_in, inp_len, pred_len, kernel_size, dilation=(1,1), order=2):
        nn.Module.__init__(self)
        self.inp_len = inp_len
        self.pred_len = pred_len
        self.order_in = order
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.c_in = c_in
        self.projection_init()

    def projection_init(self):
        kernel_size = self.kernel_size
        dilation = self.dilation
        padding = (kernel_size-1)*(dilation-1) + kernel_size -1
        self.pad_front = padding//2
        self.pad_behid = padding - self.pad_front

        inp_len = self.inp_len
        pred_len = self.pred_len
        order_in = self.order_in
        s_in = (inp_len+1)//2
        s_out = self.c_in
        n, order_in, order_out = calculate_order(self.c_in, s_in, pred_len, order_in, None)
        self.Convs = nn.ModuleList()
        self.Pools = nn.ModuleList()
        for i in range(n):
            self.Convs.append(nn.Conv2d(s_out, order_out[i]*s_out, (1,kernel_size), dilation=(1,self.dilation)))
            self.Pools.append(nn.AvgPool2d((1,order_in[i])))
            s_in = s_in // order_in[i]
            s_out = s_out * order_out[i]
        self.final_conv = nn.Conv2d(s_out,pred_len,(1,s_in))
        self.freq_layers = n

    def forward(self, x1, x2, x3):
        x1_fft = torch.fft.rfft(x1)
        x2_fft = torch.fft.rfft(x2)
        x3_fft = torch.fft.rfft(x3)
        h = torch.cat((x1_fft.imag, x2_fft.imag, x3_fft.imag,
                       x1_fft.real, x2_fft.real, x3_fft.real),dim=2)
        h = h.transpose(1,2)
        for i in range(self.freq_layers):
            h = F.pad(h,pad=(self.pad_front,self.pad_behid,0,0))
            h = self.Convs[i](h)
            h = self.Pools[i](h)
        y = self.final_conv(h).permute(0,2,3,1) + x1 + x2 + x3
        return y


class Indepent_Linear(nn.Module):
    def __init__(self, s_in, s_out, channels, share=False, dp_rate=0.5):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.randn((channels,1,s_in,s_out)))
        self.bias = nn.Parameter(torch.randn((channels,1,s_out)))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.bias)
        self.share = share
        self.dropout = nn.Dropout(dp_rate)
        if share:
            self.weight = nn.Parameter(torch.randn((1,1,s_in,s_out)))
            self.bias = nn.Parameter(torch.randn((1,1,s_out)))
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.bias)

    def forward(self, x):
        h = torch.einsum('BCNI,CNIO->BCNO',(x,self.weight))+self.bias
        return h

class stattn_layer(nn.Module):
    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps,device,
                 emb_dim=10,p=0.1,use_c_adj=False,use_c_t_adj=False):
        """
        Temporal Attention Layer
        :param num_of_vertices: int(节点数量)
        :param num_of_features: int(通道数量)
        :param num_of_timesteps: int(时间步数)
        """
        super(stattn_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.use_c_adj = use_c_adj
        self.use_c_t_adj = use_c_t_adj
        if num_of_vertices == 1:
            self.c_emb = nn.Parameter(torch.randn(num_of_features, emb_dim)) #特征嵌入
            self.adj_proj1 = nn.Parameter(torch.randn((emb_dim, emb_dim)))
            self.adj_proj2 = nn.Parameter(torch.randn((emb_dim, emb_dim)))
            nn.init.xavier_normal_(self.c_emb)
            nn.init.xavier_normal_(self.adj_proj1)
            nn.init.xavier_normal_(self.adj_proj2)

            self.neg_inf = -1e9 * torch.eye(num_of_features, device=device)

            self.c_t = nn.Parameter(torch.randn(num_of_features,num_of_timesteps)) #特征-时间步权重
            self.t_t = nn.Parameter(torch.randn(1,num_of_timesteps,num_of_timesteps)) #时间-时间矩阵
            nn.init.xavier_normal_(self.t_t)
            nn.init.xavier_normal_(self.c_t)

            self.drop = nn.Dropout(p=p)
    def forward(self,x):
        c_emb1 = torch.mm(self.c_emb, self.adj_proj1)
        c_emb2 = torch.mm(self.c_emb, self.adj_proj2)
        adj = torch.matmul(c_emb1, c_emb2.transpose(0,1))
        adj = F.relu(adj)
        adj = adj + self.neg_inf
        adj = torch.where(adj<=0.0,self.neg_inf,adj)
        adj = F.softmax(adj, dim=-1)
        x = x.squeeze(dim=2)
        x = torch.einsum("bct,cn->bnt",[x,adj]) 
        x = self.drop(x) 
        x = torch.einsum("bct,cl->btl",[x,self.c_t]) + self.t_t
        return x
