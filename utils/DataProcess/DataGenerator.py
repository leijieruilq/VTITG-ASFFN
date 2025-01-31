import torch, os, h5py
import numpy as np
import pandas as pd
from utils.DataProcess.My_DataLoader import Dataset_with_Time_Stamp

def gen_torch_dataset(data_configs, print_info=True):

    if data_configs['name'] in ['metr_la', 'pems_bay']:
        data_array, date_array = h5_file_metrla_pemsbay(data_configs, print_info)
    elif data_configs['name'] in ['pems04','pems08']:
        data_array, date_array = npz_file_pems0408(data_configs, print_info)
    elif data_configs['name'] in ['taxibj13','taxibj14','taxibj15','taxibj16']:
        data_array, date_array = h5_file_taxibj(data_configs, print_info)
    elif data_configs['name'] in ['ettm1','ettm2','etth1','etth2']:
        data_array, date_array = csv_file_ett(data_configs, print_info)
    elif data_configs['name'] in ['rate']:
        data_array, date_array = csv_file_rate(data_configs, print_info)
    elif data_configs['name'] in ['illness']:
        data_array, date_array = csv_file_ill(data_configs, print_info)
    elif data_configs['name'] in ['traffic']:
        data_array, date_array = csv_file_tra(data_configs, print_info)
    elif data_configs['name'] in ['weather']:
        data_array, date_array = csv_file_wea(data_configs, print_info)
    elif data_configs['name'] in ['electricity']:
        data_array, date_array = csv_file_ele(data_configs, print_info)
    elif data_configs['name'] in ['dsmt']:
        data_array, date_array = csv_file_dsmt(data_configs, print_info)
    else:
        raise KeyError('dataset \"{}\" not defined.'.format(data_configs['name']))

    for flag in data_configs['dataset_type']:        
        dataset = Dataset_with_Time_Stamp(
            data_array, date_array, data_configs, flag=flag)
        if not os.path.exists(data_configs['folder_path']):
            os.makedirs(data_configs['folder_path'])
        torch.save(dataset,'{}/{}.dataset'.format(data_configs['folder_path'], flag))

def csv_file_ele(data_configs, print_info):
    df = pd.read_csv(data_configs['raw_path'])
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,C = data_array.shape
    data_array = data_array.reshape(S,1,C)
    return data_array, date_array

def csv_file_dsmt(data_configs, print_info):
    df = pd.read_csv(data_configs['raw_path'])
    date_array = df['timestamp'].values[:86400]
    data_array = df.drop(columns='timestamp').values[:86400]
    S,C = data_array.shape
    data_array = data_array.reshape(S,1,C)
    print(data_array.shape)
    return data_array, date_array

def csv_file_wea(data_configs, print_info):
    df = pd.read_csv(data_configs['raw_path'])
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,C = data_array.shape
    data_array = data_array.reshape(S,1,C)
    return data_array, date_array

def csv_file_tra(data_configs, print_info):
    df = pd.read_csv(data_configs['raw_path'])
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,C = data_array.shape
    data_array = data_array.reshape(S,1,C)
    return data_array, date_array

def csv_file_rate(data_configs, print_info):
    df = pd.read_csv(data_configs['raw_path'])
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,C = data_array.shape
    data_array = data_array.reshape(S,1,C)
    return data_array, date_array

def csv_file_ill(data_configs, print_info):
    df = pd.read_csv(data_configs['raw_path'])
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,C = data_array.shape
    data_array = data_array.reshape(S,1,C)
    return data_array, date_array

def csv_file_ett(data_configs, print_info):
    df = pd.read_csv(data_configs['raw_path'])
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,C = data_array.shape
    data_array = data_array.reshape(S,1,C)
    return data_array, date_array

def h5_file_metrla_pemsbay(data_configs, print_info=True):
    try:
        with h5py.File(data_configs['rawfile_path']) as f:
            date_list = np.array(f['df']['axis1'])
            date_array = np.array([pd.Timestamp(item) for item in date_list])
            data_array = np.array(f['df']['block0_values'])
            data_array = data_array.reshape(data_array.shape[0],data_array.shape[1],1)
            f.close()
    except:
            df = pd.read_hdf(data_configs['raw_path'])
            # if you wanna use dataframe, redefine it, as the ori file is too old, and there might be some code errors.
            # attention: index might be pandas' style time stamps, that's a fking mess.
            # data = df.values
            # column = df.columns
            # index = df.index.values
            # df = pd.DataFrame(data, index=[pd.Timestamp(e) for e in index] ,columns=column)
            date_array = np.array(df.index.values)
            data_array = np.array(df.values)
            data_array = data_array.reshape(data_array.shape[0],data_array.shape[1],1)
    return data_array, date_array

def npz_file_pems0408(data_configs, print_info=True):
    data_array = np.load(data_configs['rawfile_path'])['data']
    time_list, ts_list = get_timestamp_pems0408(data_configs['start_time'], data_configs['time_window'], data_configs['lens'])
    date_array = np.array(ts_list)
    return data_array, date_array

def h5_file_taxibj(data_configs, print_info=True):
    with h5py.File(data_configs['rawfile_path']) as f:
        data_array = np.array((f['data'])).reshape(-1,2,32*32).swapaxes(1,2)
        date = np.array((f['date']),dtype=np.str_)
        f.close()
    ts_list = get_timestamp_taxibj(date)
    date_array = np.array(ts_list)
    return data_array, date_array 

def get_timestamp_pems0408(start_time, intervals, num):
    start_time = start_time.replace(':','-').replace(' ','-').split('-')
    time_list = [start_time]
    start_time = [int(item) for item in start_time]
    year = start_time[0]
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        month_list = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        month_list = [31,28,31,30,31,30,31,31,30,31,30,31]
    ts_now = '{:d}-{:d}-{:d} {:d}:{:d}:{:d}'.format(*start_time)
    ts_list = [ts_now]
    if intervals[-4:] == 'mins':
        intervals = int(intervals[:-4])
        now_time = start_time
        for i in range(num-1):
            secs = now_time[-1]
            mins = now_time[-2] + intervals
            hours = now_time[-3]
            days = now_time[-4]
            months = now_time[-5]
            years = now_time[-6]
            if secs >= 60:
                secs -= 60
                mins += 1
            if mins >= 60:
                mins -= 60
                hours += 1
            if hours >= 24:
                hours -= 24
                days += 1
            if days > month_list[months-1]:
                days -= month_list[months-1]
                months += 1
            if months > 12:
                months -= 12
                years += 1
            now_time = [years,months,days,hours,mins,secs]
            ts_now = '{:d}-{:d}-{:d} {:d}:{:d}:{:d}'.format(*now_time)
            time_list.append(now_time)
            ts_list.append(ts_now)
    return time_list,ts_list

def get_timestamp_taxibj(date):
    date_list = []
    for time in date:
        days = int(time[0:4]),int(time[4:6]),int(time[6:8])
        time_slot = int(time[8:10])
        hours = (time_slot-1) // 2
        mins = (time_slot-1) % 2
        mins = mins * 30
        date_now = '{:d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(*days,hours,mins,0)
        date_list.append(date_now)
    return date_list