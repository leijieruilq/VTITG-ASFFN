import torch.nn as nn
from .model_mav import MAVModel
from .model_mev import MEVModel

class dcaasffn_api(nn.Module):
    def __init__(self, configs, graph_generator, fixed_adjs):
        super(dcaasffn_api, self).__init__()
        model_configs = self.load_configs(configs)
        self.adjs = fixed_adjs
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        # spatial temporal predictor
        self.use_model = model_configs["use_mav"]
        if model_configs["use_mav"]:
            self.model = MAVModel(model_configs)
        else:
            self.model = MEVModel(model_configs)
 
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
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark,**args):
        if self.use_model:
            predicts, loss  = self.model(seq_x)
        else:
            predicts, loss  = self.model(seq_x)
        return predicts, loss