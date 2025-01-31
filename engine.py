import torch
import numpy as np
import torch.optim as optim
import utils.loss_box as loss_box
from utils.ExpTools import exp_summary, adjust_learning_rate, runing_confirm_file
from utils.LoadSaveTools import load_best_model, save_model, save_metrics
from utils.Logger import MyLogger, MetricsInfo, ExpInfo
from utils.ModelSummary import summary

class Basic_Engine():
    def __init__(self, configs):
        self.configs = configs
        self.set_param(configs)
        
    def set_param(self, configs):
        self.loss_rate = 1.0
        
        self.print_info = configs['info']['print_info']
        self.iter_report = configs['info']['iter_report']
        self.print_every = configs['info']['print_every']
        self.model_summary = configs['info']['model_summary']

        self.dtype = configs['envs']['dtype']
        self.device = configs['envs']['device']
        self.epochs = configs['envs']['epochs']
        self.lr=self.configs['envs']['learning_rate']
        self.adjust_lr = configs['envs']['adjust_lr']
        self.batch_size = configs['envs']['batch_size']
        self.muti_process = configs['envs']['muti_process']
        self.weight_decay=self.configs['envs']['weight_decay']
        self.load_best_model = configs['envs']['load_best_model']
        self.save_best_model = configs['envs']['save_best_model']
        self.choise_channels = torch.tensor(configs['dataset']['choise_channels'],dtype=int)
        
    def load(self, model, dataloader, dataset=None, scaler=None, **args):
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.dataloader = dataloader
        self.dataset = dataset
        self.scaler = scaler
        if 'loss' in args.keys():
            self.set_loss(args['loss'])
        if 'optimizer' in args.keys():
            self.set_opt(args['optimizer'])
        self.set_logger()
        
    def set_loss(self, info):
        if info == 'mask_mae':
            self.loss = loss_box.masked_mae
        elif info == 'mae':
            self.loss = loss_box.mae
        elif info == 'mse':
            self.loss = loss_box.mse
        elif info == 'l1':
            self.loss = torch.nn.L1Loss()
        
    def set_opt(self, info):
        if info == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def set_logger(self):
        if self.model_summary:
            for data,_,data_mark1,data_mark2 in self.dataloader['train']:
                break
            input_size1 = list(data.size()[1:])
            input_size2 = list(data_mark1.size()[1:])
            input_size3 = list(data_mark2.size()[1:])
            batch_size = self.batch_size
            try:
                model_info = summary(self.model,input_size=[input_size1,input_size2,input_size3],batch_size=batch_size, device=self.configs['envs']['device'], dtype=self.configs['envs']['dtype'])
            except Exception as e:
                model_info = 'Exception: ' + str(e)
        else:
            model_info = None
        self.logger = MyLogger(self.configs, model_info, print_while_wirte=self.print_info)
        self.metrics_saver = MetricsInfo()
        self.metrics_logger = ExpInfo()

class Engine(Basic_Engine):
    def __init__(self, configs):
        super(Engine, self).__init__(configs)
        self.running_flag = False
        
    def Run(self):
        self.metrics_logger.log_setting(self.configs) 
        for epoch in range(1, self.epochs+1):
            self.logger.start_epoch(epoch,'exp')
            for flag in ['train','vali','test']:
                metrics = self.run_epoch(epoch, flag, iter_report=self.iter_report)
                self.metrics_logger.update(epoch, metrics, flag=flag)
            exp_metrics = self.metrics_logger.get_metrics() # test report get latest 5 test loss
            latest_test_loss, summary_log = exp_summary(epoch,exp_metrics)
            lr_log = adjust_learning_rate(self.optimizer, epoch, self.adjust_lr, latest_test_loss)
            load_log = None
            if self.load_best_model:
                save_model(self.configs, self.model, epoch, self.metrics_logger, flag='epoch')
                model_dict,load_log = load_best_model(self.configs,exp_info=self.metrics_logger,current_epoch=epoch)
                if model_dict is not None:
                    self.model.load_state_dict(model_dict)
            self.logger.end_epoch(summary_log,lr_log,load_log) 
        if self.save_best_model:
            save_model(self.configs, self.model, epoch, self.metrics_logger, flag='best')
        save_metrics(self.configs, self.metrics_logger)
        self.logger.end_exp(self.metrics_logger.get_metrics())

    def run_epoch(self, epoch, flag='train', iter_report=True):
        self.metrics_saver.init_metrics()
        self.logger.start_epoch(epoch,flag,iter_report)
        for iters, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(self.dataloader[flag]): 
            loss,metrics_dict = self.run_iter(seq_x, seq_y, seq_x_mark, seq_y_mark,flag) 
            self.metrics_saver.update(loss,metrics_dict)
            if iter_report and (iters % self.print_every == 0):
                iter_log = self.metrics_saver.get_iterinfo(iters,self.print_every)
                self.logger.write_iterinfo(iter_log)
            if self.muti_process and not self.running_flag:
                runing_confirm_file(self.configs,flag='create')
                self.running_flag = True
        self.metrics_saver.update_avg()
        return self.metrics_saver.get_metrics()
    
    def run_iter(self, seq_x, seq_y, seq_x_mark, seq_y_mark, flag):
        if flag == 'train':
            self.model.train()
        else:
            self.model.eval()
        self.optimizer.zero_grad()
        output, graph_loss  = self.model(seq_x, seq_x_mark, seq_y_mark, seq_y=seq_y)
        predict,real = output,seq_y
        if self.configs["inv_transform"] == True:
            predict = self.scaler.inv_trans(output, self.choise_channels)
            real = self.scaler.inv_trans(seq_y, self.choise_channels)
        model_loss = self.loss(predict,real)
        loss = loss_box.unite_loss(model_loss, graph_loss, self.loss_rate)
        if torch.isnan(loss):
            print(None)
        metrics_dict = loss_box.metric(predict, real)
        if flag == 'train':
            loss.backward()
            self.optimizer.step()
        return loss.item(),metrics_dict