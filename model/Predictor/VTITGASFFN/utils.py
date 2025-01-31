import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    
class dynamic_adp(nn.Module):
    def __init__(self,n_channels,s_out,adp_dim,p):
        super(dynamic_adp,self).__init__()
        self.c_emb = nn.Parameter(torch.randn(n_channels,s_out,adp_dim)) #特征嵌入,考虑时间周期元素
        self.adj_proj1 = nn.Parameter(torch.randn((adp_dim,adp_dim)))
        self.adj_proj2 = nn.Parameter(torch.randn((adp_dim,adp_dim)))
        self.w = nn.Parameter(torch.randn(n_channels,s_out))
        nn.init.xavier_normal_(self.w)
        nn.init.xavier_normal_(self.c_emb)
        nn.init.xavier_normal_(self.adj_proj1)
        nn.init.xavier_normal_(self.adj_proj2)
        self.drop = nn.Dropout(p=p)

    def forward(self,x):
        c_emb1 = torch.einsum("ctd,dl->ctl",self.c_emb, self.adj_proj1)
        c_emb2 = torch.einsum("ctd,dl->ctl",self.c_emb, self.adj_proj2)
        adj = torch.einsum("ctd,ltm->clt",c_emb1,c_emb2)  #带有时间维度的动态自适应矩阵(c,c,t)
        adj = F.softmax(F.relu(adj),dim=-1)
        x = torch.einsum("bcit,cnt->bnt",[x,adj]) #融合
        x = self.drop(x)
        w_w = torch.einsum("bct,lt->blt",x,self.w) #c_c time dynamic weight forbatchsize (b,c,t)
        #np.save("adj_wea_720.npy",w_w.cpu().detach().numpy())
        #np.save("c_wea_96.npy",self.c_emb.cpu().detach().numpy())
        #np.save("c_weather_96.npy",self.c_emb.cpu().detach().numpy())
        return w_w

def multi_order(s_out, order_0, n):
        solves = []
        stats = []
        for i in range(1,int(n)):
            c = 6 * (order_0**i)
            m = n-i
            order_low = (s_out/c)**(1/(m+1)) 
            order_up = (s_out/c)**(1/m)
            order_1 = order_up//1
            if (not ((order_1 <= order_up) and (order_1 > order_low))) or (order_1 == 1):
                continue
            else:
                solves.append([order_0,order_1,i,m])
                stats.append(order_0*i+order_1*m)
        idx = np.argmin(stats)
        solves = solves[idx]
        order_list = []
        for i in range(int(n)):
            idx = np.argmax(solves[2:4])
            order_list.append(int(solves[idx]))
            solves[2+idx] -= 1
        return order_list


def calculate_order(c_in, s_in, s_out, order_in, order_out):
    n_in = (np.log(s_in)/np.log(order_in))//1 #5
    order_out_low = (s_out/c_in)**(1/(1+n_in))
    order_out_up = (s_out/c_in)**(1/(n_in))
    order_out = order_out_up//1
    n_out = (np.log(s_out/2)/np.log(order_out))//1
    if (not ((order_out <= order_out_up) and (order_out > order_out_low))) or (order_out == 1):
        Warning('Order {} is not good for s_in, s_out')
        order_out_list = multi_order(s_out, order_out, n_in)
    else:
        order_out_list = [int(order_out)]*int(n_out)
    order_in_list = [int(order_in)]*int(n_in)
    return int(n_in), order_in_list, order_out_list

class FreqConv(nn.Module):
    def __init__(self, c_in, inp_len, pred_len, kernel_size, order=2):
        nn.Module.__init__(self)
        self.inp_len = inp_len
        self.pred_len = pred_len
        self.order_in = order
        self.kernel_size = kernel_size
        self.c_in = c_in
        self.projection_init()

    def projection_init(self):
        kernel_size = self.kernel_size
        inp_len = self.inp_len
        s_in = (inp_len+1)//2
        pred_len = self.pred_len
        order_in = self.order_in
        n, order_in, order_out = calculate_order(self.c_in, s_in, pred_len, order_in, None)
        dilation = [2**(i) for i in range(n)]
        padding = [(kernel_size-1)*(dilation[i]-1) + kernel_size -1 for i in range(n)]
        self.pad_front = [padding[i]//2 for i in range(n)]
        self.pad_behid = [padding[i] - self.pad_front[i] for i in range(n)]

        s_out = self.c_in
        self.Convs = nn.ModuleList()
        self.Pools = nn.ModuleList()
        for i in range(n):
            self.Convs.append(nn.Conv2d(s_out, order_out[i]*s_out, (1,kernel_size), dilation=(1,dilation[i]),groups=s_out))
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
            h = F.pad(h,pad=(self.pad_front[i],self.pad_behid[i],0,0))
            h = self.Convs[i](h)
            h = self.Pools[i](h)
        y = self.final_conv(h).permute(0,2,3,1) + x1 + x2 + x3
        return y
