import copy,os,warnings
import numpy as np
import pandas as pd
'''
该程序用于处理实验运行后的metrics字典, 即读取root-path-result-evaluations中所有文件。
并整理为一个Excel表格, 方便观测各实验的结果。
注意修改 main 函数中的路径.
Vscode安装 Excel viewer 拓展, 可以直接查看excel文件.
'''
def main():
    names = 'all' # all表示处理所有文件，也可以填写指定文件的文件名
    path = '/home/ljr/results/evaluations2/evaluations/'
    saving_path = '' # 保存excel文件的位置，不填写表示保存至当前work space path
    excel_statistics(names,path,saving_path)

def find_the_best(dic, flag_targets, metric_target, exclude=[None], max_try=10000):
    dic = copy.deepcopy(dic)
    keys_flag = list(dic.keys())
    keys_metric = list(dic[keys_flag[0]].keys())
    if (not flag_targets in keys_flag) and (not metric_target in keys_metric):
        return None
    
    for i in range(max_try):
        best = np.min(dic[flag_targets][metric_target])
        idx = np.argmin(dic[flag_targets][metric_target])
        if not metric_target in ['loss','mape','rmse']:
            if best in exclude:
                dic[flag_targets][metric_target][idx] = np.inf
            else:
                break
        else:
            n_iters = len(dic[flag_targets][metric_target][0])
            idx1 = idx // n_iters
            idx2 = idx - idx1 * n_iters
            idx = (idx1,idx2)
            if best in exclude:
                dic[flag_targets][metric_target][idx1][idx2] = np.inf
            else:
                break
    return best,idx,i

def get_title(metrics):
    # 'Notes',
    title = ['Notes', 
             'Min Vali MAE Idx','Avg. MAE','Avg. MAPE','Avg. RMSE','Avg. RMSE_2']
    title = title + ['15mins MAE','15mins MAPE','15mins RMSE','15mins RMSE_2',
                     '30mins MAE','30mins MAPE','30mins RMSE','30mins RMSE_2',
                     '60mins MAE','60mins MAPE','60mins RMSE','60mins RMSE_2']
    title = title + list(metrics['hyperparameters'].keys())
    title = title + ['Min Train MAE','Min Vali MAE','Min Test MAE']
    title = title + ['Current MAE 12 Step'] + list(range(1,12+1))
    title = title + ['Current MAPE 12 Step'] + list(range(1,12+1))
    title = title + ['Current RMSE 12 Step'] + list(range(1,12+1))
    title = title + ['Current RMSE_2 12 Step'] + list(range(1,12+1))
    return title

def get_statistics(metrics):
    min_train_mae = np.min(metrics['train']['m_mae'])
    min_vali_idx = np.argmin(metrics['vali']['m_mae'])
    min_vali_mae = np.min(metrics['vali']['m_mae'])
    min_test_mae = np.min(metrics['test']['m_mae'])

    test_mae_idxed = metrics['test']['m_mae'][min_vali_idx]
    test_mape_idxed = metrics['test']['m_mape'][min_vali_idx] * 100
    test_rmse_idxed = metrics['test']['m_rmse'][min_vali_idx]
    test_rmse_2_idxed = metrics['test']['m_rmse_2'][min_vali_idx]
    # metrics['notes']
    MAE_12step = list(metrics['test']['m_mae_all'][min_vali_idx])
    MAPE_12step = list(metrics['test']['m_mape_all'][min_vali_idx])
    RMSE_12step = list(metrics['test']['m_rmse_all'][min_vali_idx])
    RMSE_2_12step = list(metrics['test']['m_rmse_all_2'][min_vali_idx])
    statistics = [metrics['notes'],
                  min_vali_idx+1,
                  test_mae_idxed, test_mape_idxed, test_rmse_idxed, test_rmse_2_idxed,
                  MAE_12step[2],MAPE_12step[2],RMSE_12step[2],RMSE_2_12step[2],
                  MAE_12step[5],MAPE_12step[5],RMSE_12step[5],RMSE_2_12step[5],
                  MAE_12step[11],MAPE_12step[11],RMSE_12step[11],RMSE_2_12step[11]] +\
                 [v[0] for v in metrics['hyperparameters'].values()] +\
                 [min_train_mae,min_vali_mae,min_test_mae] +\
                 [min_vali_idx+1] + MAE_12step[:12] +\
                 [min_vali_idx+1] + MAPE_12step[:12] +\
                 [min_vali_idx+1] + RMSE_12step[:12] +\
                 [min_vali_idx+1] + RMSE_2_12step[:12]
    return statistics

def excel_statistics(files,path,saving_path):
    print('Processing.')
    files_name = os.listdir(path) if files  == 'all' else files
    files_name.sort()
    flag = True
    this_path = os.path.dirname(os.path.abspath(__file__))
    if saving_path == '':
        saving_path = this_path + '/'
    saving_name = 'metrics_info.xlsx'
    writer = pd.ExcelWriter(saving_path+saving_name)		# 写入Excel文件
    for i,name in enumerate(files_name):
        obj = np.load(path + name, allow_pickle = True)
        metrics = obj.item()
        print('\t {:02d}/{:02d}: <{:s}>.'.format(i+1,len(files_name),name))
        # '''Train Validation Test mae every epoch'''
        index = [name, '', '']
        train_mape = [
            metrics['train']['m_mae'],
            metrics['vali']['m_mae'],
            metrics['test']['m_mae']]
        train_mape = pd.DataFrame(train_mape, index=index, columns=list(range(1,len(metrics['train']['m_mape'])+1)))
        if flag:
            train_mape.to_excel(writer, 'page_1', startrow=0)
        else:
            train_mape.to_excel(writer, 'page_1', startrow=1+(i*3), header=None)		# ‘page_1’是写入excel的sheet名

        # '''Statistics Infomation'''
        title = get_title(metrics)
        # state_info
        statistics_info = get_statistics(metrics)
        statistics_info = pd.DataFrame([statistics_info], index=[name], columns=title)
        if flag:
            statistics_info.to_excel(writer, 'page_2', startrow=0 ,float_format='%.15f')
        else:
            statistics_info.to_excel(writer, 'page_2', startrow=1+i, header=None ,float_format='%.15f')
        flag = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        writer._save()
    writer.close()
    print('\tSaving Path: \'{:s}\'.\nDone.'.format(saving_path+saving_name))

if __name__  == '__main__':
    main()