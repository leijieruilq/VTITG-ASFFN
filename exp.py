import os
# setting GPU Index.
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import argparse
from engine import Engine
from utils.ExpTools import my_bool, setting_seed
import datetime 
from utils.ConfigLoader import load_configs
from utils.DataProcess.DataProvider import DataProcessor
from utils.ModelProvider import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--exp_configs_path', type=str, default='./configs/Exp/engine.yaml', help='engine settings')
parser.add_argument('--dataset_configs_path', type=str, default='./configs/Dataset/dataset.yaml', help='dataset info')
parser.add_argument('--model_configs_path', type=str, default='./configs/Models/', help='model settings')
parser.add_argument('--graphgen_configs_path', type=str, default='./configs/GraphGenerator/', help='settings of graph generator')
parser.add_argument('--root_path', type=str, default='./', help='Root directory for reading and writing data')
parser.add_argument('--model_name', type=str, default='vtitg-asffn', help='model name: tcaffn,vtitg-asffn')
parser.add_argument('--dataset_name', type=str, default='traffic', help='dataset name: ett(h1,h2,m1,m2), metr_la, pems_bay, pems04, pems08, taxibj13 (13-16)'+
                    'electricity,rate,illness,traffic,weather,dsmt')
parser.add_argument('--graphgen_name', type=str, default='None', help='None: not use')
parser.add_argument('--choise_channels', type=list, default=[-1], help='choose channels, -1: all, [0,1,...]')
parser.add_argument('--regen_dataset', type=my_bool, default=True, help='Dataloader is regenerated with every run.')
parser.add_argument('--move_step', type=int, default=13, help='')
parser.add_argument('--loss', type=str, default='l1', help='')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='')
parser.add_argument('--weight_decay', type=float, default=0, help='')

parser.add_argument('--dataset_prob', type=list, default=[0.7,0.1,0.2], help='[Train, validation, test].')
parser.add_argument('--data_scale', type=my_bool, default=True, help='Whether or not to normalize the data.')
parser.add_argument('--date_scale', type=my_bool, default=True, help='Max-Min normalization.')
parser.add_argument('--c_out', type=int, default=-1, help='The number of channels to be included in the prediction, -1 means the same as the input channels, the number of input channels is determined by choice_channels.')
parser.add_argument('--check_dataloader', type=my_bool, default=False, help='NAN check')
parser.add_argument('--device', type=str, default='cuda:1', help='cpu, cuda. cuda:0-n')
parser.add_argument('--epochs', type=int, default=42, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--inp_len', type=int, default=96, help='input length')
parser.add_argument('--pred_len', type=int, default=96, help='predict length')
parser.add_argument('--optimizer', type=str, default='amam', help='adam')
parser.add_argument('--save_log', type=my_bool, default=True)
parser.add_argument('--print_info', type=my_bool, default=True)
parser.add_argument('--iter_report', type=my_bool, default=True)
parser.add_argument('--print_every', type=int, default=100, help='Outputs a Loss report for every number of iter hours.')
parser.add_argument('--model_summary', type=my_bool, default=False, help='Output information such as model results and parameter counts before each task')
parser.add_argument('--seed', type=int, default=42, help='')
parser.add_argument('--inv_transform', type=my_bool, default=False, help='Whether to do the inverse normalization or not, default no (not done for calculating the loss in the paper)')
args = parser.parse_args()

def main(configs):
    dataloader, dataset, scaler, _ = DataProcessor(configs).get_data()
    model = get_model(configs)
    exp = Engine(configs)
    exp.load(model, dataloader, dataset, scaler, loss=configs['envs']['loss'], optimizer='adam')
    configs['muti_process'] = False
    exp.Run()

if __name__ == '__main__':
    setting_seed(args.seed)
    start_time = datetime.datetime.now()  
    configs = load_configs(args).get_configs()
    main(configs)
    end_time = datetime.datetime.now()  
    runtime = end_time - start_time 
    print(f"Script finished. Total runtime: {runtime.total_seconds()} seconds")