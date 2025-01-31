import torch,os,json
import numpy as np
from utils.ExpTools import check_dict_item

def save_model(configs, model, epoch, exp_info, flag):
    # saving model state.
    saving_path = '{}model_state/{}/{}/{}/'.format(configs['path']['saving_path'], flag, configs['dataset']['name'], configs['info']['exp_start_time'])
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    task_name = '' if configs['info']['task_name'] is None else '_' + configs['info']['task_name']
    name = 'Epoch_{}-Loss_{}{}'.format(str(epoch),str(round(exp_info.metrics_info['vali']['m_loss'][epoch-1],2)),task_name)
    torch.save(model.state_dict(), saving_path + name + '.model')

def load_best_model(configs, exp_info, current_epoch, flag='epoch'):
    # pass loading
    if current_epoch == configs['envs']['epochs']:
        return (None,None)
    # find best model state.
    bestid = np.argmin(list(exp_info.metrics_info['test']['m_loss']))
    best_epoch = bestid + 1
    if best_epoch == current_epoch:
        return (None,None)
    # find path and state name
    saving_path = '{}model_state/{}/{}/{}/'.format(configs['path']['saving_path'], flag, configs['dataset']['name'], configs['info']['exp_start_time'])
    task_name = '' if configs['info']['task_name'] is None else '_' + configs['info']['task_name']
    name = 'Epoch_{}-Loss_{}{}'.format(str(best_epoch),str(round(exp_info.metrics_info['vali']['m_loss'][best_epoch-1],2)),task_name)
    model = torch.load(saving_path + name + '.model')
    log ='\n\t<loading best validation model: {}>'.format(name)
    return model, log

def save_metrics(configs, dic):
    # saving metrics result, type: dict.
    path = '{}evaluations/'.format(configs['path']['saving_path'])
    task_name = '' if configs['info']['task_name'] is None else '_' + configs['info']['task_name']
    name = '{}{}'.format(configs['info']['exp_start_time'], task_name)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save('{}{}.npy'.format(path,name), dic.get_metrics())


