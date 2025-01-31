import torch, os, copy
from torch.utils.data import DataLoader
from utils.DataProcess.DataGenerator import gen_torch_dataset
from utils.DataProcess.AdjProvider import get_adj

class DataProcessor():
    def __init__(self, configs):
        self.configs = configs
        self.file_path = configs['path']['file_path']
        self.raw_path = configs['path']['raw_path']
        self.device = configs['envs']['device']
        self.dtype = configs['envs']['dtype']
        self.batch_size = configs['envs']['batch_size']
        self.print_info = configs['info']['print_info']
        self.check_dataloader_bool = configs['envs']['check_dataloader']
    
    def get_data(self):
        self.find_gen_dataloader(self.configs,regen=self.configs['envs']['regen_dataset'])
        dataloader, dataset, scaler = self.load_files(self.configs['dataset'])
        if self.check_dataloader_bool:
            self.test_dataloader(dataloader)
        adjs = get_adj(path=self.configs['dataset']['adj_path'], n_nodes=self.configs['dataset']['n_nodes']).to(device=self.device,dtype=self.dtype)
        return dataloader, dataset, scaler, adjs
    
    def test_dataloader(self, dataloader):
        # print("Testing dataloader.")
        for key in ['train','test','vali']:
            for item in dataloader:
                pass

    def find_gen_dataloader(self, configs, regen=False):
        data_configs = configs['dataset']

        dataloader_bool = True
        for flag in ['train','test','vali']:
            dataset_path = '{}/{}.dataset'.format(data_configs['folder_path'],flag)
            dataloader_bool = dataloader_bool and os.path.exists(dataset_path)
        
        if (not dataloader_bool) or regen:
            if self.print_info:
                if regen:
                    print('Regenerating dataset.')
                elif not dataloader_bool:
                    print('Target dataset \"{}\" Not finded, Generating.'.format(data_configs['folder_path']))
                
            if not os.path.exists(data_configs['folder_path']):
                os.makedirs(data_configs['folder_path'])
            
            data_configs['dataset_type'] = ['train','test','vali']
            data_configs['scale'] = True
            data_configs['period_type'] = 'week'
            data_configs['inp_len'] = configs['envs']['inp_len']
            data_configs['pred_len'] = configs['envs']['pred_len']
            data_configs['dtype'] = configs['envs']['dtype']
            data_configs['num_nodes'] = configs['dataset']['n_nodes']
            data_configs['n_channels'] = len(configs['dataset']['channel_info'])
            data_configs['data_scale'] = configs['envs']['data_scale']
            data_configs['date_scale'] = configs['envs']['date_scale']
            data_configs['dataset_prob'] = configs['envs']['dataset_prob']

            gen_torch_dataset(data_configs, self.print_info)
            if self.print_info:
                print('\tDone.')
            
    def load_files(self, data_configs):
        if self.print_info:
            print('Loading existence torch dataset.')
        dataset = {'train':None,'vali':None,'test':None}
        dataloader = {'train':None,'vali':None,'test':None}
        
        channels_info = [data_configs['channel_info'][idx] for idx in data_configs['choise_channels']]

        if self.print_info:
            print('\tchoice channels {}'.format(channels_info))
        for flag in ['train','vali','test']:
            # path
            file_path = '{}/{}.dataset'.format(data_configs['folder_path'], flag)
            data_set = torch.load(file_path)
            data_set.choice(data_configs['choise_channels'])
            data_set.to(self.device, self.dtype)
            # parameter
            batch_size = self.batch_size
            drop_last = True
            shuffle = False if flag == 'test' else True
            # load
            if self.print_info:
                print('\t{:5}: {}'.format(flag, len(data_set)))
            dataset[flag] = data_set
            dataloader[flag] = DataLoader(data_set,
                                    batch_size = batch_size,
                                    shuffle = shuffle,
                                    drop_last = drop_last)
        # scaler
        scaler = dataset['train'].scaler.to(self.device)
        return dataloader, dataset, scaler