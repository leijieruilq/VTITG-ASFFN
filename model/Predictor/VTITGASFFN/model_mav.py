import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import FreqConv, Indepent_Linear, gated_mlp, dynamic_adp

class hyperparameters(nn.Module):
    def __init__(self, configs) -> None:
        nn.Module.__init__(self)
        self.n_nodes = configs['n_nodes']
        self.c_date = configs['c_date']

        self.c_in = configs['c_in']
        self.c_out = configs['c_out']
        self.s_in = configs['inp_len']
        self.device = configs['device']
        self.n_channels = configs['c_in']
        self.s_out = configs['pred_len']
        self.dp_rate = configs['dropout']
        
        self.adp_dim = configs['adp_dim']
        self.channels_emb_layers = configs['layers']
        self.share = configs['share']
        self.use_update = configs["use_update"]
        self.use_guide = configs["use_guide"]

class MAVModel(hyperparameters):
    def __init__(self, configs, **args):
        hyperparameters.__init__(self, configs)
        self.fconv1 = FreqConv(6, self.s_in, self.s_in, kernel_size=3, order=2)
        self.fconv2 = FreqConv(6, self.s_out, self.s_out, kernel_size=3, order=2)
        self.fc_idp = Indepent_Linear(self.s_in, self.s_out, self.n_channels, self.share, self.dp_rate)

        self.dadp_in = nn.ModuleList([dynamic_adp(n_channels=self.n_channels,
                                                  s_out=self.s_in,
                                                  adp_dim=self.adp_dim,p=self.dp_rate) for i in range(self.channels_emb_layers)])
        
        self.gated_mlps_in = nn.ModuleList([gated_mlp(seq_in=self.s_in,seq_out=self.s_in,channels=self.n_channels,
                                                      use_update=self.use_update
                                                      ) for i in range(self.channels_emb_layers)])
        
        self.dadp_out = nn.ModuleList([dynamic_adp(n_channels=self.n_channels,
                                                    s_out=self.s_out,
                                                    adp_dim=self.adp_dim,p=self.dp_rate) for i in range(self.channels_emb_layers)])
        
        self.gated_mlps_out = nn.ModuleList([gated_mlp(seq_in=self.s_out,seq_out=self.s_out,channels=self.n_channels,
                                                      use_update=self.use_update
                                                      ) for i in range(self.channels_emb_layers)])
        self.mask_in = nn.Parameter(torch.rand(1, self.n_channels, 1, int(self.s_in/2)+1)) #自适应mask
        nn.init.xavier_normal_(self.mask_in)

        self.mask_out = nn.Parameter(torch.rand(1, self.n_channels, 1, int(self.s_out/2)+1)) #自适应mask
        nn.init.xavier_normal_(self.mask_out)
        
    def freq_attn_in(self, x):
        freq = torch.fft.rfft(x)
        y = torch.fft.irfft((self.mask_in)*freq)
        return y
    
    def freq_attn_out(self, x):
        freq = torch.fft.rfft(x)
        y = torch.fft.irfft((self.mask_out)*freq)
        return y

    def forward(self, x):
        b,c,n,t = x.shape
        if self.use_guide:
            x_t = self.freq_attn_in(x)
        else:
            x_t = torch.zeros((b,self.c_in,1,self.s_in),device=x.device)
        x_c = x + x_t 
        for (layer,mlp) in zip(self.dadp_in,self.gated_mlps_in):
            x_c = mlp(x_c)
            x_c = torch.einsum("bcnt,blt->blnt",[x_c,F.gelu(layer(x_c))]) + x_c
        h_x = self.fconv1(x, x_t, x_c)
        h_y = self.fc_idp(h_x)
        if self.use_guide:
            y_t = self.freq_attn_out(h_y)
        else:
            y_t = torch.zeros((b,self.c_in,1,self.s_out),device=x.device)
        y_c = h_y + y_t
        for (layer,mlp) in zip(self.dadp_out,self.gated_mlps_out):
            y_c = mlp(y_c) 
            y_c = torch.einsum("bcnt,blt->blnt",[y_c,F.gelu(layer(y_c))]) + y_c
        y = self.fconv2(h_y, y_t, y_c)
        loss = 0.0
        return y, loss