import datetime
from txgnn import TxData, TxGNN, TxEval
import builtins
import time

def print(*args, **kwargs):
    builtins.print(*args, **kwargs, flush=True)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f'[{get_timestamp()}] Importing libraries ...')

data_path = '/om/user/tysinger/kg/'

print(f'[{get_timestamp()}] Loading data ...')

TxData_inst = TxData(data_folder_path = data_path)
TxData_inst.prepare_split(split = 'random', seed = 42, no_kg = False)
print(f'[{get_timestamp()}] Loaded data!')
print(f'[{get_timestamp()}] Initializing model ...')

TxGNN_model = TxGNN(data = TxData_inst, 
                weight_bias_track = False,
                proj_name = 'TxGNN',
                exp_name = 'TxGNN',
                device = 'cuda:0'
                )
"""
TxGNN_model.model_initialize(n_hid = 1280, 
                        n_inp = 1280, 
                        n_out = 1280, 
                        proto = False, #made this False
                        proto_num = 3,
                        attention = False,
                        sim_measure = 'all_nodes_profile',
                        bert_measure = 'disease_name',
                        agg_measure = 'rarity',
                        num_walks = 200,
                        walk_mode = 'bit',
                        path_length = 2,
                        esm = True)
print(f'[{get_timestamp()}] Initialized model!')

print(f'[{get_timestamp()}] Starting to pretrain ...')
TxGNN_model.pretrain(save_path = '/om/user/tysinger/models', 
                     name = 'pretrain_esm', 
                     n_epoch = 3, 
                     learning_rate = 1e-3,
                     batch_size = 1024, 
                     train_print_per_n = 500,
                     save_per_n = 1)
print(f'[{get_timestamp()}] Pretrain done! ')

TxGNN_model.save_model('/om/user/tysinger/models/pretrain_esm')
"""
TxGNN_model.load_pretrained('/om/user/tysinger/models/pretrain_esm_1')
print(f'[{get_timestamp()}] Starting to finetune ...')
TxGNN_model.finetune(save_path = '/om/user/tysinger/models', 
                     name = '150finetune_esm',
                     n_epoch = 150, 
                     learning_rate = 5e-4,
                     train_print_per_n = 50,
                     valid_per_n = 5,
                     save_per_n = 10)
print(f'[{get_timestamp()}] Finetune done! ')

TxGNN_model.save_model('/om/user/tysinger/models/esm_finetuned150')

