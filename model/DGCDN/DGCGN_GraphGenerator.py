import torch
import torch.nn as nn

from utils.loss_box import Graph_Contrastive_Loss
from model.Universal.Embedding import Graph_seq_Embedding
from model.DGCDN.DGCDN_GraphMemoryBlock import Graph_LSTM,Graph_GRU
from model.DGCDN.Time_series_similarity import time_delay_self_attn,dist_measurements
from model.DGCDN.My_functional import muti_heads_attn,denoising_filter_channels,linear_as_conv2d,scale_hard_sigmoid,scale_hard_tanh

class Graph_Generator(nn.Module):
    def __init__(self, configs, topo_graph):
        super(Graph_Generator, self).__init__()
        self.configs = configs
        # '''feature graph'''
        if configs['local_obs']:
            self.feat_graphs_generator = feature_graphs_generator(configs)
        # '''probability graph'''
        self.prob_graphs_generator = probability_graph_generator(configs,topo_graph)
        # '''projection'''
        self.embedding = Graph_seq_Embedding(configs) # input projection
        self.feat_filter_bool = configs['feat_filter'] * configs['local_obs']
        self.prob_filter_bool = configs['prob_filter']
        if self.feat_filter_bool:
            self.feat_filter = denoising_filter_channels(configs, configs['n_feat'], configs['n_feat'], filter_type=1)
        if self.prob_filter_bool:
            self.prob_filter = denoising_filter_channels(configs, configs['n_prob'], configs['n_prob'], filter_type=1)
    
    def forward(self,seq_x,seq_x_marks):
        x = self.embedding(seq_x,seq_x_marks)
        # '''P & N Obeservations'''                                                     # np.save('saved_graph/simi_feat',simi_feat_graphs.cpu().detach().numpy()) 
        simi_feat_graphs, dist_feat_graphs, feat_CL_loss = self.feat_graphs_generator(x) if self.configs['local_obs'] else (None, None, 0.0)
        # '''Probability Distribution Learning'''                                       # np.save('saved_graph/diff_feat',dist_feat_graphs.cpu().detach().numpy())
        if self.feat_filter_bool:
            simi_feat_graphs = self.feat_filter(x, simi_feat_graphs, dist_feat_graphs)
        posi_prob_graph, nega_prob_graph, prob_CL_loss = self.prob_graphs_generator(simi_feat_graphs,dist_feat_graphs)
        # '''Denoising Filter'''                                                        # np.save('saved_graph/posi_prob',posi_prob_graph.cpu().detach().numpy())
        if self.prob_filter_bool:
            real_graph = self.prob_filter(x,posi_prob_graph,nega_prob_graph)            # np.save('saved_graph/nega_prob',nega_prob_graph.cpu().detach().numpy())
        else:
            real_graph = posi_prob_graph
        # '''Graph contrastive loss'''                                                  # np.save('saved_graph/real_graph',real_graph.cpu().detach().numpy())
        CL_loss = feat_CL_loss +  prob_CL_loss
        if torch.isnan(CL_loss):
            print('hear')
        return real_graph, CL_loss

class feature_graphs_generator(nn.Module):
    def __init__(self, configs):
        super(feature_graphs_generator,self).__init__()
        self.heads = configs['n_feat']
        seq_len = configs['seq_len']
        simi_type = configs['simi_type']
        dist_type = configs['dist_type']
        channels = configs['n_channels']
        hid_dim = configs['attn_hid_dim']
        # '''contrastive loss'''
        self.ND_GC_loss = Graph_Contrastive_Loss(configs['local_obs'], 
                                                 disturb_indices_bool=False, indices_disturb_type='samble_only')
        # '''functional'''
        # simi graph
        if simi_type == 'attn':
            self.simi_generator = muti_heads_attn(seq_len, channels, hid_dim, self.heads,
                                                  activation=nn.Sigmoid(), dropout=configs['dropout'])
        elif simi_type == 'TDA':
            self.simi_generator = time_delay_self_attn(seq_len, channels, self.heads,
                                                       activation=nn.Tanh(), abs=True, device=configs['device'])
        self.simi_activation = nn.Sigmoid()
        # dist graph
        if dist_type == 'attn':
            self.dist_generator = muti_heads_attn(seq_len, channels, hid_dim, self.heads,
                                                  activation=nn.Sigmoid(), dropout=configs['dropout'])
        elif dist_type == 'dist':
            self.dist_generator = dist_measurements(seq_len, channels, self.heads,
                                                    device=configs['device'])
        # CL_loss
        self.CL_loss_bool = configs['feat_cl_loss']
        if self.CL_loss_bool:
            self.CL_loss = Graph_Contrastive_Loss(configs['sparity_rate'],
                                                  disturb_indices_bool=False, indices_disturb_type='samble_only')
    
    def forward(self, x):
        simi_graph = self.simi_generator(x)
        dist_graph = 1 - self.dist_generator(x)
        if self.CL_loss_bool:
            feat_CL_loss = self.CL_loss(simi_graph, dist_graph, self.training)
        else:
            feat_CL_loss = 0.0
        return simi_graph, dist_graph, feat_CL_loss

class probability_graph_generator(nn.Module):
    def __init__(self, configs, topo_graph):
        super(probability_graph_generator,self).__init__()
        # '''positive & negative probability graph'''
        self.posi_prob_block = probability_model(configs,topo_graph,attribute='+')
        self.nega_prob_block = probability_model(configs,topo_graph,attribute='-')
        self.CL_loss_bool = configs['prob_cl_loss']
        if self.CL_loss_bool:
            self.CL_loss = Graph_Contrastive_Loss(cl_sparity_rate=configs['sparity_rate'],
                                                    disturb_indices_bool=False, indices_disturb_type='samble_only')

    def forward(self, posi_feat_graph, nega_feat_graph):
        posi_prob_graph = self.posi_prob_block(posi_feat_graph)
        nega_prob_graph = self.nega_prob_block(nega_feat_graph)
        if self.CL_loss_bool:
            prob_CL_loss = self.CL_loss(posi_prob_graph,nega_prob_graph,self.training)
        else:
            prob_CL_loss = 0.0
        return posi_prob_graph, nega_prob_graph, prob_CL_loss

class probability_model(nn.Module):
    def __init__(self,configs,topo_graph,attribute):
        super(probability_model,self).__init__()
        self.configs = configs
        self.attribute = attribute
        # '''hyper parameters'''
        self.gamma_lower,self.gamma_upper = configs['gamma']
        # '''graph memory block'''
        topo_graph = topo_graph if attribute == '+' else 1-topo_graph
        if configs['memory_type'] == 'LSTM':
            self.LSTM_Memory = Graph_LSTM(configs,topo_graph)
        elif configs['memory_type'] == 'GRU':
            self.GRU_Memory = Graph_GRU(configs,topo_graph)
        elif configs['memory_type'] == 'LPM':
            if configs['LPM_learnable']:
                self.learnable_matrix = nn.Parameter(topo_graph.to(configs['device']))
            else:
                self.learnable_matrix = topo_graph.to(configs['device'])
        # '''projection'''
        if self.configs['local_obs'] and (not self.configs['global_obs']):
            c_in = self.configs['n_feat']
            c_out = self.configs['n_feat'] if configs['global_obs'] else self.configs['n_prob']
            self.feat_graph_fc = linear_as_conv2d(c_in, c_out) # initiating as an identity mapping
        elif (self.configs['local_obs'] or self.configs['global_obs']):
            c_in = self.configs['n_feat'] + self.configs['n_GMB'] if configs['local_obs'] else self.configs['n_GMB']
            c_out = self.configs['n_prob']
            self.prob_graph_fc = linear_as_conv2d(c_in, c_out)
        # '''functional'''
        self.scaling_activate = scale_hard_tanh if attribute == '+' else scale_hard_sigmoid
    
    def add_noise(self,graph):
        if self.training:# only training
            eps = self.configs['graph_noise']
            graph = graph + eps*torch.randn_like(graph)
        return graph
    
    def LPM(self):
        global_view = self.add_noise(self.learnable_matrix)                 # np.save('saved_graph/posi_LPM',self.learnable_matrix.cpu().detach().numpy())
        global_view = global_view.unsqueeze(0).repeat(self.configs['batch_size'],1,1,1)  # np.save('saved_graph/nega_LPM',self.learnable_matrix.cpu().detach().numpy())
        return global_view
    
    def Graph_LSTM(self,feat_graph):
        global_view = self.LSTM_Memory(feat_graph)
        # global_view = self.add_noise(global_view)
        return global_view

    def Graph_GRU(self,feat_graph):
        global_view = self.GRU_Memory(feat_graph)
        # global_view = self.add_noise(global_view)
        return global_view

    def forward(self, local_view):
        # '''global view observation'''
        if self.configs['memory_type'] == 'LPM':
            global_view = self.LPM()
        elif self.configs['memory_type'] == 'LSTM':
            global_view = self.Graph_LSTM(local_view)
        elif self.configs['memory_type'] == 'GRU':
            global_view = self.Graph_GRU(local_view)
        # '''probability graph infference'''
        if self.configs['local_obs'] and self.configs['global_obs']:
            prob_graph = torch.cat((global_view, local_view),dim=1)
            prob_graph = self.prob_graph_fc(prob_graph)
        elif self.configs['global_obs']:
            prob_graph = self.prob_graph_fc(global_view)
        elif self.configs['local_obs']:
            prob_graph = self.feat_graph_fc(local_view)
        else:
            prob_graph = torch.eye(self.configs['n_nodes']).unsqueeze(0).\
                unsqueeze(1).repeat(self.configs['batch_size'],self.configs['n_prob'],1,1).\
                to(self.configs['device'])
        prob_graph = self.scaling_activate(prob_graph,self.gamma_lower,self.gamma_upper,activate=True)
        return prob_graph