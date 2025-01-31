import time, os, multiprocessing, copy, torch

from engine import Engine
from utils.ConfigLoader import load_configs

from utils.ExpTools import check_dict_item,setting_seed,get_device_info,runing_confirm_file
import argparse
from engine import Engine
from utils.ExpTools import my_bool, setting_seed

from utils.ConfigLoader import load_configs
from utils.DataProcess.DataProvider import DataProcessor
from utils.ModelProvider import get_model, get_graphgenerator
import sys

class MutiProcess_tasks():
    def __init__(self, program_args, max_waiting=20*5, max_retry=6*12, waiting_time=3, retring_time=60*10):
        time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
        print('{}| Main Class PID: {}'.format(time_now, os.getpid()))
        self.program_args = program_args
        self.max_waiting = max_waiting
        self.max_retry = max_retry
        self.waiting_time = waiting_time
        self.retring_time = retring_time
    
    def RUN(self, task_dict):
        process_list, configs_list, task_load_flag_list, task_running_flag_list = self.init_tasks(task_dict)
        self.load_tasks(process_list, configs_list, task_load_flag_list, task_running_flag_list)
        for p in process_list:
            if p.is_alive():
                time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                print('{}|\tWaiting for joining, name: \"{}\", PID: {}.'.format(time_now,p.name,p.pid))
                p.join()
            else:
                time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                print('{}|\tTask finished, name: \"{}\".'.format(time_now,p.name))
        time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
        print('{}| Taks all Done.'.format(time_now))
    
    def init_tasks(self, task_dict):
        # muti process tasks list
        process_list = []
        configs_list = []
        task_load_flag_list = []
        task_running_flag_list = []

        for task_name in list(task_dict.keys()):
            task_configs = self.gen_new_args(self.program_args, task_dict[task_name])
            task_configs['info']['task_name'] = task_name
            task_configs['task_name'] = task_name

            process_list.append(multiprocessing.Process(name=task_name, target=self.run_task, args=list(task_configs.items())))
            configs_list.append(task_configs)
            task_load_flag_list.append(0)
            task_running_flag_list.append(0)
        return process_list, configs_list, task_load_flag_list, task_running_flag_list
    
    def gen_new_args(self, args, task_configs):
        configs = vars(args)
        configs.update(task_configs)
        args = copy.deepcopy(argparse.Namespace(**configs))
        configs = load_configs(args).get_configs()
        return copy.deepcopy(configs)
    
    def run_task(self, *configs):
        configs = dict(copy.deepcopy(configs))
        setting_seed(configs['envs']['seed']) # 42: all the truths of the universe
        
        error_flag = False
        # if configs['model']['name']=='arima':
        #     dataloader, _, scaler, _ = DataProcessor(configs).get_data()
        #     model = get_model(configs)
        #     model.run(dataloader["test"],scaler)
        #     return
        # 加载数据集
        dataloader, dataset, scaler, fixed_adjs = DataProcessor(configs).get_data()
        # 加载图生成器
        graph_generator = get_graphgenerator(configs, fixed_adjs)
        # 获取模型
        model = get_model(configs, graph_generator, fixed_adjs)
        # 初始化实验
        exp = Engine(configs)
        # 设定特殊的参数（其实没啥用）
        if not configs['graphgenerator'] is None:
            if configs['graphgenerator']['name'] == 'CLGSDN':
                exp.loss_rate = configs['graphgenerator']['loss_rate']
        # 向实验框架中加载模型及数据
        exp.load(model, dataloader, dataset, scaler, loss=configs['envs']['loss'], optimizer='adam')
        
        try:
            # 实验！启动！
            exp.Run()
        except Exception as e:
            time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
            print('{}|\tTask \"{}\" fail to start, info: \"{}\".'.format(time_now, configs['info']['task_name'], str(e)))
            error_flag = True
            runing_confirm_file(configs, flag='create', state='fail')
        if not error_flag:
            time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
            print('{}| End task: {}.'.format(time_now,configs['info']['task_name']))

    def load_tasks(self, process_list, configs_list, task_load_flag_list, task_running_flag_list):
        retry_times = 0
        PID_list = [os.getpid()]
        while True:
            # all task loaded confirm
            if sum(task_load_flag_list) == len(task_load_flag_list):
                time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                print('{}| All Task Loaded.'.format(time_now))
                print("{}| PID: ".format(time_now), end='')
                for _pid in PID_list:
                    print('{} '.format(_pid),end='')
                print('')
            if sum(task_running_flag_list) == len(task_running_flag_list):
                time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                print('{}| All Task Running.'.format(time_now))
                break
            # load task
            unfinish_tasks = [task_running_flag_list[i] * task_load_flag_list[i] for i in range(len(task_load_flag_list))]
            for i,item in enumerate(unfinish_tasks):
                if item >= 1:
                    continue
                # task info
                task = process_list[i]
                task_configs = configs_list[i]
                task_name = task.name
                # try to load
                time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                print('{}| Try to load Task: \"{}\".'.format(time_now,task_name))
                # get free memorty
                device = task_configs['envs']['device']
                _,_,freemem = get_device_info(device)
                # loading
                error_type = None
                try:
                    if freemem > self.program_args.least_memory:
                        time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                        print('{}|\tDevice: \"{}\", Memory free: {} MB.'.format(time_now,device,freemem))
                        time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                        print('{}|\tTaks notes: \"{}\".'.format(time_now,task_configs['info']['notes']))
                        try:
                            task.start()
                        except Exception as e_2:
                            error_type = 'StartTaskERROR'
                            raise e_2
                        time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                        print('{}|\tTask started, PID: {}.'.format(time_now, task.pid))
                        # task loaded confirm
                        task_load_flag_list[i] = 1
                        # taks running confirm
                        for _ in range(self.max_waiting):
                            if runing_confirm_file(task_configs, flag='check', state='fail'):
                                task_load_flag_list[i] = 0
                                time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                                print('{}|\tTask error confirmed.'.format(time_now))
                                runing_confirm_file(task_configs,'del','fail')
                                raise RuntimeError('Run time error.')
                            elif runing_confirm_file(task_configs, flag='check'):
                                _,_,freemem_now = get_device_info(device)
                                time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                                print('{}|\tTask running confirmed, usage memory: {} MB.'.format(time_now,freemem-freemem_now))
                                task_running_flag_list[i] = 1
                                PID_list.append(task.pid)
                                break
                            time.sleep(self.waiting_time)
                        if task_running_flag_list[i] == 0:
                            error_type = 'RunningConfirmERROR'
                            raise RuntimeError('Failed to confirm task running: \"{}\".'.format(task_name))
                        time.sleep(self.waiting_time)
                    else:
                        raise torch.cuda.OutOfMemoryError('Out of Memory.')
                except Exception as e:
                    time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                    print('{}| ---Task Error Information---'.format(time_now))
                    if error_type is None:
                        time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                        print('{}|\tinfo: \"{}\".'.format(time_now,e))
                    elif error_type == 'RunningConfirmERROR':
                        time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                        print('{}|\tinfo: \"Failed to confirm task running\".'.format(time_now))
                    elif error_type == 'StartTaskERROR':
                        time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                        print('{}|\tinfo: \"Failed to start task\".'.format(time_now))
                    
                    time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                    print('{}|\tName: \"{}\".'.format(time_now,task_name))
                    # reinitiate task
                    process_list[i] = None
                    process_list[i] = multiprocessing.Process(name=task_name, target=self.run_task, args=list(task_configs.items()))
                    # retry counte
                    retry_times += 1
                    if retry_times > self.max_retry:
                        time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                        print('{}|---Processor failed to finish all the tasks---'.format(time_now))
                        return
                    time_now = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                    print('{}|\tRetry times: {}/{}. waiting time: {} s.'.format(time_now,retry_times,self.max_retry,self.retring_time))
                    time.sleep(self.retring_time)