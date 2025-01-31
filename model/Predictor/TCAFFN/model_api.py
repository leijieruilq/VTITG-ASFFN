import torch.nn as nn
from .model import Model

class fcn_api(nn.Module):
    def __init__(self, configs, graph_generator, fixed_adjs):
        super(fcn_api, self).__init__()
        model_configs = self.load_configs(configs)
        self.adjs = fixed_adjs
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        self.model = Model(model_configs)
 
    def load_configs(self, configs):
        model_configs = configs['model']
        model_configs['c_date'] = configs['dataset']['c_date']
        model_configs['n_nodes'] = configs['dataset']['n_nodes']
        model_configs['c_in'] = configs['envs']['c_in']
        model_configs['c_out'] = configs['envs']['c_out']
        model_configs['device'] = configs['envs']['device']
        model_configs['inp_len'] = configs['envs']['inp_len']
        model_configs['pred_len'] = configs['envs']['pred_len']
        return model_configs
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        predicts, loss  = self.model(seq_x, seq_x_mark, seq_y_mark, adjs=None)
        return predicts, loss