import torch
import torch.nn as nn
from model.DGCDN.My_functional import *

class Graph_GRU(nn.Module):
    def __init__(self,configs,init_graph):
        super(Graph_GRU,self).__init__()
        self.configs = configs
        self.batch_size = configs['batch_size']
        # projection
        c_in = 2*configs['n_feat']
        c_out = configs['n_GMB']
        self.fc_z = nn.Linear(c_in,c_out)
        self.fc_r = nn.Linear(c_in,c_out)
        self.fc_w = nn.Linear(c_in,c_out)
        # functional
        self.dropout2d = nn.Dropout2d(configs['dropout'])
        self.init_state(init_graph)
    
    def init_state(self,init_graph):
        self.h_hid = (init_graph.to(self.configs['device']).unsqueeze(0)).permute(0,2,3,1)

    def update_memory(self,input_x):
        h = self.h_hid.repeat(self.batch_size,1,1,1)
        # update gate
        z = torch.cat((h,input_x),dim=-1)
        z = F.sigmoid(self.fc_z(z))
        # reset gate
        r = torch.cat((h,input_x),dim=-1)
        r = F.sigmoid(self.fc_r(r))
        # candidate state
        h_hat = torch.cat((r*h,input_x),dim=-1)
        h_hat = F.tanh(self.fc_w(h_hat))
        # new hidden state
        h_new = (1-z)*h + z*h_hat
        return h_new
    
    def forward(self, graph):
        h = self.update_memory(graph.permute(0,2,3,1))
        if self.training:
            self.h_hid = h.mean(0).unsqueeze(0).data
        h = h.permute(0,3,1,2)
        return h
    
class Graph_LSTM(nn.Module):
    def __init__(self,configs,init_graph):
        super(Graph_LSTM,self).__init__()
        self.configs = configs
        self.batch_size = configs['batch_size']
        # projection
        c_in = 2*configs['n_feat']
        c_out = configs['n_GMB']
        self.fc_f = nn.Linear(c_in,c_out)
        self.fc_i = nn.Linear(c_in,c_out)
        self.fc_o = nn.Linear(c_in,c_out)
        self.fc_c = nn.Linear(c_in,c_out)
        # functional
        self.dropout = nn.Dropout(configs['dropout'])
        self.dropout2d = nn.Dropout2d(configs['dropout'])
        self.init_state(init_graph)
        
    def init_state(self,init_graph):
        self.h_hid = (init_graph.to(self.configs['device']).unsqueeze(0)).permute(0,2,3,1)
        self.cell = (init_graph.to(self.configs['device']).unsqueeze(0)).permute(0,2,3,1)
    
    def update_memory(self,feat_graph):
        h_hid = self.h_hid.repeat(self.batch_size,1,1,1)
        cell = self.cell.repeat(self.batch_size,1,1,1)
        h_tilde = torch.cat((h_hid,feat_graph),dim=-1)
        # forget gate
        f = self.dropout(F.sigmoid(self.fc_f(h_tilde))) 
        # input gate
        i = self.dropout(F.sigmoid(self.fc_i(h_tilde)))
        # output gate
        o = self.dropout(F.sigmoid(self.fc_o(h_tilde)))
        # cell state
        c_tilde = F.tanh(self.dropout(self.fc_c(h_tilde)))
        cell_new = f*cell + i*c_tilde
        # h output
        h_new = o*F.tanh(cell_new)            
        return h_new,cell_new
    
    def forward(self,graph):
        h,cell = self.update_memory(graph.permute(0,2,3,1))
        if self.training:
            self.cell = cell.mean(0).unsqueeze(0).data
            self.h_hid = h.mean(0).unsqueeze(0).data
        h = h.permute(0,3,1,2)
        return h