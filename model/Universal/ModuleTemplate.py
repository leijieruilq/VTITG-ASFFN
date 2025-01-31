
import torch.nn as nn

class My_Module(nn.Module):
    def __init__(self, configs):
        super(My_Module, self).__init__()
        self.configs = configs
        self.device = configs['envs']['device']
        self.dtype = configs['envs']['dtype']
        self.batch_size = configs['envs']['batch_size']
        self.using_adjs = configs['model']['using_adjs']
        self.batch_size = configs['envs']['batch_size']
        self.using_graph_generator = False
        
    def load_fixed_adjs(self, adjs):
        self.fixed_adjs = adjs.to(self.device, self.dtype)
        self.fixed_adjs_loaded = True
    
    def load_graph_generator(self, graph_generator):
        self.graph_generator = graph_generator
        self.using_graph_generator = True
        
    def load_st_model(self, model):
        if self.using_adjs:
            assert self.fixed_adjs is not None or self.graph_generator is not None
        self.model = model

    def get_adjs(self, seq_x, seq_x_mark):
        graph_loss = 0.0
        if not self.using_adjs:
            return None, graph_loss
        if self.using_graph_generator:
            adjs, graph_loss = self.graph_generator(seq_x, seq_x_mark)
            return adjs, graph_loss
        return self.fixed_adjs, graph_loss