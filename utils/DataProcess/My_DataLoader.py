import torch, warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class Dataset_with_Time_Stamp(Dataset):
    '''
    自定义的dataset
    该class初始化时提供  data_array: 数据的numpy array
                        date_array: 数据的时间标签
    '''
    def __init__(self, data_array, date_array, data_configs, flag):
        self.data = data_array
        self.date = date_array
        self.set_parameters(data_configs, flag)
        self.__read_data__()

    def set_parameters(self, configs, flag):    
        self.num_nodes = configs['num_nodes']
        self.n_channels = configs['n_channels']
        self.period_type = configs['period_type']
        self.inp_len = configs['inp_len']
        self.pred_len = configs['pred_len']
        self.seq_len = self.inp_len +self.pred_len
        # init
        assert flag in ['train', 'test', 'vali']
        type_map = {'train': 0, 'vali': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_scale = configs['data_scale']
        self.date_scale = configs['date_scale']
        self.freq = configs['time_window']

        self.dtype = configs['dtype']
        self.dataset_prob = configs['dataset_prob']
        self.choice_channels = configs['choise_channels']

    def get_border_period(self):
        if self.freq[-4:] == 'mins':
            freq = float("".join(list(filter(str.isdigit, self.freq))))
            if self.period_type == 'week':
                period_len = self.data.shape[0]/ (60 / freq * 24 * 7)
                num_unit_period = int(60/freq * 24 * 7)
            elif self.period_type == 'month':
                period_len = self.data.shape[0]/ (60 / freq * 24 * 30)
                num_unit_period = int(60/freq * 24 * 30)
            assert (sum(self.dataset_prob) - 1) < 1e-6
            test_period = int(period_len * self.dataset_prob[-1])
            val_period = int(period_len * self.dataset_prob[-2])
            train_period = period_len - test_period - val_period
            border1s = np.array([
                0,
                train_period,
                (train_period + val_period)], dtype=int)* num_unit_period
            border2s = np.array([
                train_period,
                train_period + val_period,
                train_period + test_period + val_period], dtype=int) * num_unit_period
            return border1s, border2s, freq
        else:
            raise ValueError('dataset for freq \"{}\" not define.'.format(self.freq))
    
    def get_border(self):
        if self.freq[-4:] == 'mins':
            freq = float("".join(list(filter(str.isdigit, self.freq))))
            data_len = self.data.shape[0]
            assert sum(self.dataset_prob) == 1
            test_len = int(data_len * self.dataset_prob[-1])
            val_len = int(data_len * self.dataset_prob[-2])
            train_len = data_len - test_len - val_len
            border1s = np.array([
                0,
                train_len,
                (train_len + val_len)],dtype=int)
            border2s = np.array([
                train_len,
                train_len + val_len,
                train_len + val_len + test_len],dtype=int)
            return border1s,border2s, freq
        else:
            raise ValueError('dataset for freq \"{}\" not define.'.format(self.freq))
    
    def __read_data__(self):
        if self.period_type is not None:
            border1s,border2s,freq = self.get_border_period()
        else:
            border1s,border2s,freq = self.get_border()
        
        tf_data = torch.tensor(self.data, dtype=self.dtype)
        if self.data_scale: 
            train_data = tf_data[border1s[0]: border2s[0]]
            self.scaler = DataScaler(self.choice_channels)
            _,N,C = train_data.size()
            assert N == self.num_nodes and C == self.n_channels, 'Tensor axis error: make sure that your axis is [n time windows, nodes, channels]'
            # or you can still run your program but get a confusing model performace.
            self.scaler.fit(train_data)
            data = self.scaler.trans(tf_data)
        else:
            self.scaler = None
            data = tf_data

        data_stamp = pd.DataFrame(self.date, columns=['date'])
        data_stamp['date'] = pd.to_datetime(data_stamp['date'])
        warnings.filterwarnings('ignore')
        data_stamp['month'] = data_stamp['date'].apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp['date'].apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp['date'].apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp['date'].apply(lambda row: row.hour, 1)
        data_stamp['minute'] = data_stamp['date'].apply(lambda row: row.minute, 1)
        data_stamp['minute'] = data_stamp['minute'].map(lambda x: x // freq)
        data_stamp = data_stamp.drop(columns='date').values
        data_stamp = torch.tensor(data_stamp,dtype=self.dtype)

        if self.date_scale: 
            train_data = data_stamp[border1s[0]:border2s[0]]
            self.stamp_scaler = Stamp_DataScaler()
            self.stamp_scaler.fit(train_data)
            data_stamp = self.stamp_scaler.trans(data_stamp)
        else:
            self.stamp_scaler = None

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.data = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]
  
    def choice(self, channels):
        self.data = self.data[:,:,torch.tensor(channels)]
    
    def to(self, device='cpu',dtype=None):
        if dtype is None:
            dtype = self.dtype
        self.scaler.to(device)
        self.stamp_scaler.to(device)
        self.data = self.data.to(device=device,dtype=dtype)
        self.data_stamp = self.data_stamp.to(device=device,dtype=dtype)
        return 
    
    def __getitem__(self, index):
        begin = index
        med = begin + self.inp_len
        end = begin + self.seq_len
        seq_x = self.data[begin:med]
        seq_y = self.data[med:end]
        seq_x_mark = self.data_stamp[begin:med]
        seq_y_mark = self.data_stamp[med:end]
        return seq_x.permute(2,1,0), seq_y.permute(2,1,0), seq_x_mark, seq_y_mark
    
    def _set_dtype(self,dtype):
        self.data = self.data.to(dtype)
        self.data_stamp = self.data_stamp.to(dtype)

    def __len__(self):
        return len(self.data) - self.seq_len
    
class Stamp_DataScaler():
    def __init__(self):
        # Max Min Normalization
        self.max = 0
        self.min = 0

    def to(self, device):
        # self.to(device)
        self.max = self.max.to(device)
        self.min = self.min.to(device)
        return self
    
    def fit_trans(self, data):
        self.fit(data)
        return self.trans(data)
    
    def fit(self, data):
        self.max = data.max((0))[0].unsqueeze(0)
        self.min = data.min((0))[0].unsqueeze(0)
        for i in range(data.size(-1)):
            if self.max[...,i] == self.min[...,i]:
                self.max[...,i] = 0.0
                self.min[...,i] = self.max[...,i]-1

    def trans(self,data):
        data = (data - self.min)/(self.max-self.min)
        return data

    def inverse_transform(self, data, choise_channels):
        data = (self.max-self.min)*data + self.min
        return data

class DataScaler():
    def __init__(self, choice_channels):
        self.mean = 0
        self.std = None
        self.channels = choice_channels

    def to(self, device):
        # self.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.max_v_ori = self.max_v_ori.to(device)
        self.min_v_ori = self.min_v_ori.to(device)
        self.max_v_trans = self.max_v_trans.to(device)
        self.min_v_trans = self.min_v_trans.to(device)
        return self

    def fit_trans(self,data):
        self.fit(data)
        return self.trans(data)

    def fit(self, data):
        # be sure that your channels is [time, nodes, channels]
        self.mean = data.mean((0,1))
        self.std = data.std((0,1))
        self.flag = 1
        self.max_v_ori = data.max(dim=0)[0].max(dim=0)[0]
        self.min_v_ori = data.min(dim=0)[0].min(dim=0)[0]
        self.max_v_trans = self.trans(data).max(dim=0)[0].max(dim=0)[0]
        self.min_v_trans = self.trans(data).min(dim=0)[0].min(dim=0)[0]

    def trans(self,data):
        data = (data - self.mean.unsqueeze(0).unsqueeze(1))/self.std.unsqueeze(0).unsqueeze(1)
        return data
    
    def inv_trans(self, data, choise_channels):
        # [Batch, channels, nodes, time windows]
        std = self.std[self.channels].reshape(1,len(choise_channels),1,1)
        mean = self.mean[self.channels].reshape(1,len(choise_channels),1,1)
        data = data * std + mean
        return data
    
    def trans_MaxMin(self, data):
        min_v = self.min_v_trans[self.channels].reshape(1,len(self.channels),1,1)
        max_v = self.max_v_trans[self.channels].reshape(1,len(self.channels),1,1)
        data = (data - min_v)/(max_v - min_v)
        return data
    
    def inv_trans_MaxMin(self, data):
        min_v = self.min_v_trans[self.channels].reshape(1,len(self.channels),1,1)
        max_v = self.max_v_trans[self.channels].reshape(1,len(self.channels),1,1)
        data = data*(max_v - min_v)+min_v
        return data
    
    def trans_MaxMin_ori(self, data):
        min_v = self.min_v_ori[self.channels].reshape(1,len(self.channels),1,1)
        max_v = self.max_v_ori[self.channels].reshape(1,len(self.channels),1,1)
        data = (data - min_v)/(max_v - min_v)
        return data

    def inv_trans_MaxMin_ori(self, data):
        min_v = self.min_v_ori[self.channels].reshape(1,len(self.channels),1,1)
        max_v = self.max_v_ori[self.channels].reshape(1,len(self.channels),1,1)
        data = data*(max_v - min_v)+min_v
        return data