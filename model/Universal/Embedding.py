import torch,math
import torch.nn as nn
import numpy as np

class ChannelEmbedding(nn.Module):
    def __init__(self,c_in,c_out):
        super(ChannelEmbedding,self).__init__()
        self.channel_embed = nn.Conv2d(c_in,c_out,(1,1))

    def forward(self,inputs):
        x_embed = self.channel_embed(inputs)
        return x_embed
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x)
    
class Graph_seq_Embedding(nn.Module):
    def __init__(self, configs):
        super(Graph_seq_Embedding, self).__init__()
        # self.FixEmbedding = nn.Parameter(torch.zeros((1, configs['n_nodes'], configs['seq_len'])))
        self.dropout = nn.Dropout(configs['dropout'])
        # Embedding
        self.fix_emb = nn.Parameter(torch.randn((1,configs['n_channels'],configs['n_nodes'],configs['seq_len'])))        
        self.posi_emb = PositionalEmbedding(configs['seq_len'],configs['n_nodes'])

        # self.date_emb = TemporalEmbedding(configs['n_channels'])
        self.seq_fc = nn.Linear(configs['seq_len'],configs['seq_len'])
        self.date_emb = nn.Linear(configs['dim_date'],configs['n_nodes'])
        self.channel_emb = nn.Conv2d(configs['data_channels'], configs['n_channels'], (1,1))

    def forward(self, seq_x, seq_x_marks):
        # date embedding
        x_date = self.date_emb(seq_x_marks).transpose(1,2).unsqueeze(1)
        # position embedding
        x_posi = self.posi_emb(seq_x).unsqueeze(1)
        # channel embedding
        x_seq_mapped = self.seq_fc(seq_x)
        x_token = self.channel_emb(x_seq_mapped)
        # delay_emb = self.Delay_emb(seq_x, seq_x_marks, Memory_block_in)
        delay_emb = 0.0
        # outputs
        x = (x_date + x_posi + x_token)/3.0 + (self.fix_emb + delay_emb)
        return x
    
# class Graph_seq_Embedding(nn.Module):
#     def __init__(self, configs):
#         super(Graph_seq_Embedding, self).__init__()
#         # self.FixEmbedding = nn.Parameter(torch.zeros((1, configs['n_nodes'], configs['seq_len'])))
#         self.dropout = nn.Dropout(configs['dropout'])
#         self.fix_emb = nn.Parameter(torch.randn((1,configs['n_nodes'],configs['seq_len'])))
#         self.seq_fc = nn.Linear(configs['seq_len'],configs['seq_len'])
#         self.date_fc = nn.Linear(configs['dim_date'],configs['n_nodes'])
#         self.posi_emb = PositionalEmbedding(configs['seq_len'],configs['n_nodes'])

#     def forward(self, seq_x, seq_x_marks):
#         x1 = self.seq_fc(seq_x.transpose(1,2))
#         x2 = self.date_fc(seq_x_marks).transpose(1,2)
#         x3 = self.posi_emb(seq_x.transpose(1,2))
#         x = (x1+x2+x3)/3.0 + self.fix_emb
#         return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, n_position=1024):
        super(PositionalEmbedding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_model))

    def _get_sinusoid_encoding_table(self, n_position, d_model):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(2)].clone().detach()

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self,n_channels) -> None:
        super(TemporalEmbedding,self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        self.minute_embed = nn.Embedding(minute_size,n_channels)
        self.hour_embed = nn.Embedding(hour_size,n_channels)
        self.weekday_embed = nn.Embedding(weekday_size,n_channels)
        self.day_embed = nn.Embedding(day_size,n_channels)
        self.month_embed = nn.Embedding(month_size,n_channels)
    
    def forward(self,inputs):
        inputs = inputs.to(torch.long)
        min_out = self.minute_embed(inputs[:,:,4])
        hour_out = self.hour_embed(inputs[:,:,3])
        weekday_out = self.weekday_embed(inputs[:,:,2])
        day_out = self.day_embed(inputs[:,:,1])
        month_out = self.month_embed(inputs[:,:,0])
        out = min_out + hour_out + weekday_out + day_out + month_out
        return out
    
# class TemporalEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='fixed', freq='h'):
#         super(TemporalEmbedding, self).__init__()

#         minute_size = 4
#         hour_size = 24
#         weekday_size = 7
#         day_size = 32
#         month_size = 13

#         Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
#         if freq == 't':
#             self.minute_embed = Embed(minute_size, d_model)
#         self.hour_embed = Embed(hour_size, d_model)
#         self.weekday_embed = Embed(weekday_size, d_model)
#         self.day_embed = Embed(day_size, d_model)
#         self.month_embed = Embed(month_size, d_model)

#     def forward(self, x):
#         x = x.long()

#         minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
#         hour_x = self.hour_embed(x[:, :, 3])
#         weekday_x = self.weekday_embed(x[:, :, 2])
#         day_x = self.day_embed(x[:, :, 1])
#         month_x = self.month_embed(x[:, :, 0])

#         return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
    
