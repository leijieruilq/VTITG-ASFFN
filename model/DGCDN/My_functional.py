import torch
import torch.nn as nn
import torch.nn.functional as F

class muti_heads_attn(nn.Module):
    def __init__(self, seq_len, channels, hid_dim, heads=1, activation=None, dropout=None):
        super(muti_heads_attn,self).__init__()
        # hyper params
        self.heads = heads
        # projection
        self.fc_q = nn.Linear(seq_len*channels, hid_dim*heads)
        self.fc_k = nn.Linear(seq_len*channels, hid_dim*heads)
        # functinal
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.activation = activation

    def forward(self,x):
        B,C,N,S = x.size()
        x = x.transpose(1,2).reshape(B,N,-1)
        Q = self.fc_q(x)
        K = self.fc_k(x)
        if self.dropout is not None:
            Q = self.dropout(Q)
            K = self.dropout(K)
        Q = Q.reshape(B,N,self.heads,-1).transpose(1,2)
        K = K.reshape(B,N,self.heads,-1).transpose(1,2)
        corrs = torch.matmul(Q,K.transpose(2,3))
        if self.activation is not None:
            corrs = self.activation(corrs)
        return corrs

class denoising_filter_channels(nn.Module):
    def __init__(self,configs,c_in,c_out,filter_type=1):
        super(denoising_filter_channels,self).__init__()
        seq_len, n_channels, hid_dim, dropout = configs['seq_len'], configs['n_channels'], configs['attn_hid_dim'], configs['dropout']
        # '''hyper parameters'''
        self.gamma_lower, self.gamma_upper = configs['gamma']
        self.heads = c_out
        self.filter_type = filter_type
        # '''functional'''
        self.dropout = nn.Dropout(dropout)
        self.graph_filter = self.type1_filter if filter_type == 1 else self.type2_filter
        # '''projection'''
        self.filter_fc = linear_as_conv2d(c_in, c_out, dropout=configs['dropout'])
        self.heads = c_out
        self.attn_bool = configs['filter_attn']
        if configs['filter_attn']:
            self.attn_fc = muti_heads_attn(seq_len, n_channels, hid_dim, c_out)
    
    def type1_filter(self,x,posi_graph,nega_graph):
        # filter_inp = self.filter_fc(1-nega_graph)
        filter_inp = self.filter_fc(1-nega_graph)
        if self.attn_bool:
            filter_refine = self.attn_fc(x)
            filter_ = SHA_filter_type1(filter_inp, filter_refine, self.gamma_lower, self.gamma_upper)
        else:
            filter_refine = None
            filter_ = scale_hard_sigmoid(filter_inp, self.gamma_lower, self.gamma_upper)
        filted_graph = filter_ * posi_graph
        return filted_graph
    
    def type2_filter(self,x,posi_graph,nega_graph):
        if self.attn_bool:
            filter_refine = self.attn_fc(x)
            filter_inp = torch.cat((nega_graph,filter_refine),dim=1)
        else:
            filter_refine = None
        filter_inp = torch.cat((nega_graph,filter_refine),dim=1)
        filter_inp = self.filter_fc(filter_inp)
        filter_ = SHA_filter_type2(filter_inp, None, self.gamma_lower, self.gamma_upper)
        filted_graph = filter_ * posi_graph
        return filted_graph
    
    def forward(self,x,posi_graph,nega_graph):
        return self.graph_filter(x,posi_graph,nega_graph)
    
def SHA_filter_type1(x1,x2,low=-0.0,up=1.0,activate=True):
    if activate:
        x1 = F.sigmoid(x1)
        x2 = 1 - F.sigmoid(x2)
    else:
        x2 = 1-x2
    x1 = x1*(up-low)+low
    x2 = x2*(up-low)+low
    x = x1 * x2
    x = 1-F.relu(1-F.relu(x))
    return x

def SHA_filter_type2(x,low=-0.0,up=1.0,activate=True):
    if activate:
        x = F.sigmoid(1-x)
    x = x*(up-low)+low
    x = 1-F.relu(1-F.relu(x))
    return x

def scale_hard_sigmoid(x,low=-0.0,up=1.0,activate=True):
    if activate:
        x = F.sigmoid(x)
    x = x*(up-low)+low
    x = 1-F.relu(1-F.relu(x))
    return x

def scale_hard_tanh(x,low=-0.0,up=1.0,activate=True):
    if activate:
        x = F.tanh(x)
    x = (x+1)/2*(up+1-low)+low-1
    x = 1-F.relu(2-F.relu(x+1))
    return x

class linear_as_conv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=None, init_self=None, bias=True, dropout=None):
        '''Nobody like permute in the model.'''
        # 1*1 conv + bias = linear
        super(linear_as_conv2d,self).__init__()

        self.fc = nn.Linear(c_in,c_out,bias=bias)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if init_self is not None:
            if init_self == 'ones':
                self.ones_init()
            elif init_self == 'identity':
                self.identity_init()

    def ones_init(self):
        self.fc.weight.data = torch.ones_like(self.fc.weight.data)/self.fc.weight.data.size(1)
        self.fc.bias.data = torch.zeros_like(self.fc.bias.data) # initiating as an identity mapping

    def identity_init(self):
        assert self.fc.weight.data.size(0) == self.fc.weight.data.size(1)
        self.fc.weight.data = torch.eye(self.fc.weight.data.size(0)) # initiating as an identity mapping
        self.fc.bias.data = torch.zeros_like(self.fc.bias.data)
    
    def forward(self,x):
        x = self.fc(x.permute(0,2,3,1))
        if self.dropout is not None:
            x = self.dropout(x)
        return x.permute(0,3,1,2)
    
class linear_as_conv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=None, bias=True, dropout=None):
        # 1*1 conv + bias = linear
        super(linear_as_conv1d,self).__init__()
        
        self.fc = nn.Linear(c_in,c_out,bias)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
    
    def forward(self,x):
        
        x = self.fc(x.transpose(-2,-1))

        if self.dropout is not None:
            x = self.dropout(x)
        
        return x.transpose(-2,-1)

