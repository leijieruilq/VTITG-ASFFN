import os, torch, time, copy, yaml

class load_configs():
    def __init__(self, args):
        self.set_params(args)
        configs = self.read_file(args)
        configs = self.map_dict_item(configs)
        self.print_info = configs['info']['print_info']
        new_configs = {}
        new_configs['inv_transform'] = configs['inv_transform']
        new_configs['dataset'] = self.reset_dataset(configs['dataset'], args)
        new_configs['info'] = self.reset_info(configs['info'])
        new_configs['envs'] = self.reset_envs(configs['envs'])
        new_configs['path'] = self.reset_path(configs['path'])
        new_configs['model'] = self.reset_model(configs['model'])

        new_configs = self.reset_overall(new_configs)
        self.configs = new_configs
    
    def set_params(self, args):
        self.map_dict = {'': None,
                         'None':None,
                         'torch.float':torch.float,
                         'False':False,
                         'F':False,
                         'f':False,
                         'True':True,
                         'T':True,
                         't':True
                         }
        self.map_keys = list(self.map_dict.keys())
        self.args_configs = vars(args)
        self.args_keys = self.args_configs.keys()

    def map_dict_item(self, configs):
        configs = copy.deepcopy(configs)
        new_configs = {}
        dict_keys = configs.keys()
        for key in dict_keys:
            # if type is dict, continue recursion
            if type(configs[key]) == dict:
                new_configs[key] = self.map_dict_item(configs[key])
            elif key in self.args_keys:
                value = self.args_configs[key]
                if value in self.map_keys:
                    new_configs[key] = self.map_dict[value]
                else:
                    new_configs[key] = value
            elif configs[key] in self.map_keys:
                new_configs[key] = self.map_dict[configs[key]]
            else:
                new_configs[key] = configs[key]
            if type(new_configs[key]) == str:
                # for scientific notation 'xey': x*10^(y) -> float
                if 'e' in new_configs[key]:
                    try:
                        new_configs[key] = float(new_configs[key])
                    except:
                        pass
        return copy.deepcopy(new_configs)
    
    def read_file(self,args):
        configs = {}
        # Engine
        with open(args.exp_configs_path) as f:
            configs.update(yaml.safe_load(f))
        # Dataset
        with open(args.dataset_configs_path) as f:
            configs['dataset'] = yaml.safe_load(f)
        # Model
        model_configs_path = '{}{}.yaml'.format(args.model_configs_path, args.model_name)
        with open(model_configs_path) as f:
            configs['model'] = yaml.safe_load(f)
        configs['inv_transform'] = args.inv_transform
        return configs
    
    def get_configs(self):
        return self.configs

    def reset_info(self, configs):
        # experiments information
        configs['exp_ID'] = os.getpid()
        configs['exp_start_time'] = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
        return copy.deepcopy(configs)
        
    def reset_envs(self, configs):
        # check cuda
        if not torch.cuda.is_available():
            if self.print_info:
                print('Cuda not available. device -> cpu')
            configs['device'] = torch.device('cpu')
        else:
            configs['device'] = torch.device(configs['device'])
        return copy.deepcopy(configs)

    def reset_dataset(self, configs, args):
        dataset_name = args.dataset_name
        if not dataset_name in configs.keys():
            raise NameError("Cannot Find Dataset \"{}\".".format(dataset_name))
        configs = configs[dataset_name]
        return copy.deepcopy(configs)

    def reset_path(self, configs):
        if configs['raw_path'] == '':
            configs['raw_path'] = './'
        configs['raw_path'] = configs['root_path'] + configs['raw_path']
        configs['file_path'] = configs['root_path'] + configs['file_path']
        configs['saving_path'] = configs['root_path'] + configs['saving_path']
        return copy.deepcopy(configs)
    
    def reset_graphgenerator(self, configs):
        return copy.deepcopy(configs)

    def reset_model(self, configs):
        return copy.deepcopy(configs)

    def reset_overall(self, configs):
        configs['dataset']['raw_path'] = configs['path']['raw_path'] + configs['dataset']['folder_path'] + configs['dataset']['file_name']
        configs['dataset']['folder_path'] = '{}{}'.format(configs['path']['file_path'], configs['dataset']['name'])
        if configs['dataset']['adj_path'] is not None:
            configs['dataset']['adj_path'] = '{}{}/{}'.format(configs['path']['raw_path'], configs['dataset']['name'], configs['dataset']['adj_path'])
        all_channels = list(range(len(configs['dataset']['channel_info'])))
        if configs['dataset']['choise_channels'] == [-1]:
            configs['dataset']['choise_channels'] = all_channels
        for item in configs['dataset']['choise_channels']:
            if item not in all_channels:
                raise ValueError('Params \"choise_channels\" shoule not \"{}\".'.format(configs['dataset']['choise_channels']))
        configs['dataset']['data_channels'] = len(configs['dataset']['choise_channels'])
        configs['envs']['c_in'] = len(configs['dataset']['choise_channels'])
        if configs['envs']['c_out'] == -1:
            configs['envs']['c_out'] = configs['envs']['c_in']
        return copy.deepcopy(configs)
