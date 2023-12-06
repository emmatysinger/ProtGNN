import datetime
from txgnn import TxData, TxGNN, TxEval
import builtins
import time
import pandas as pd
import argparse
import wandb


def main(args):
    def print(*args, **kwargs):
        builtins.print(*args, **kwargs, flush=True)

    def get_timestamp():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_path = '/om/user/tysinger/kg/'

    print(f'[{get_timestamp()}] Loading data ...')
    TxData_inst = TxData(data_folder_path = data_path)
    TxData_inst.prepare_split(split = 'random', seed = 42, no_kg = False)
    print(f'[{get_timestamp()}] Loaded data!')
    print(f'[{get_timestamp()}] Initializing model ...')

    TxGNN_model = TxGNN(data = TxData_inst, 
                    weight_bias_track = True,
                    proj_name = 'MEng',
                    exp_name = 'ESM Finetune LR 0.005',
                    device = 'cuda:0'
                    )
    
    if args.pretrain:
        sweep_configuration = {
            "method": "bayes",
            "name": "sweep",
            "metric": {"goal": "minimize", "name": "validation_loss"},
            "parameters": {
                "batch_size": {"values": [512, 1024, 2048]},
                "epochs": {"values": [1,2,3]},
                "lr": {"max": 0.1, "min": 0.0001},
            },
        }

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

    if args.finetune:
        if args.pretrain:
            TxGNN_model = TxGNN(data = TxData_inst, 
                    weight_bias_track = True,
                    proj_name = 'MEng',
                    exp_name = 'ESM Finetune',
                    device = 'cuda:0'
                    )
        TxGNN_model.load_pretrained('/om/user/tysinger/models/pretrain_esm_1')

        print(f'[{get_timestamp()}] Starting to finetune ...')
        TxGNN_model.finetune(save_path = '/om/user/tysinger/models', 
                            name = 'esm_finetuned',
                            n_epoch = 150, 
                            learning_rate = 5e-3,
                            train_print_per_n = 5,
                            valid_per_n = 10,
                            save_per_n = 1000,
                            b=0.2)
        print(f'[{get_timestamp()}] Finetune done! ')
        TxGNN_model.save_model('/om/user/tysinger/models/esm_finetunedLRFlood')

    if args.eval:
        TxGNN_model.load_pretrained('/om/user/tysinger/models/random_finetuned150')
        print(f'[{get_timestamp()}] Starting evaluation ...')

        TxEval_model = TxEval(model = TxGNN_model)
        #TxEval_model.load_eval_results(save_name = 'random_sigmoid_eval.pkl') 
        #TxEval_model.eval()


        #df_test = pd.read_csv('/om/user/tysinger/kg/random_42/test.csv')
        #filtered_df = df_test[df_test.x_type == 'molecular_function']
        #random_node_idxs = [int(i) for i in list(filtered_df.sample(n=200)['x_idx'])]


        result = TxEval_model.eval_molfunc_centric(molfunc_idxs = 'test_set', #'test_set'
                                            show_plot = 'random_eval_finetune_update', 
                                            verbose = True, 
                                            save_result = True,
                                            save_name = 'random_eval_finetune_update.pkl',
                                            return_raw = False)
        print(result)

    if args.hyperparameter_tuning:
        def train():
            wandb.init(project="my-first-sweep")
            print(wandb.config)
            return None
            TxGNN_model = TxGNN(data = TxData_inst, 
                    weight_bias_track = False,
                    proj_name = 'MEng',
                    exp_name = 'Sweep_Test',
                    device = 'cuda:0'
                    )

            print('Initializing Model')
            TxGNN_model.model_initialize(n_hid = wandb.config.n_hid, 
                            n_inp = wandb.config.n_inp, 
                            n_out = wandb.config.n_out, 
                            proto = False, #made this False
                            proto_num = 3,
                            attention = False,
                            sim_measure = 'all_nodes_profile',
                            bert_measure = 'disease_name',
                            agg_measure = 'rarity',
                            num_walks = 200,
                            walk_mode = 'bit',
                            path_length = 2)
            
            TxGNN_model.load_pretrained('/om/user/tysinger/models/pretrain_random_1')
            print('Finetuning Model')
            TxGNN_model.finetune(n_epoch = wandb.config.epochs, 
                    learning_rate = wandb.config.lr,
                    train_print_per_n = 5,
                    valid_per_n = 50,
                    save_path = '/om/user/tysinger/models', 
                    name = 'sweep')
        
        sweep_configuration = {
            "method": "bayes",
            "name": "sweep_test",
            "metric": {"goal": "minimize", "name": "validation_loss"},
            "parameters": {
                "n_hid": {"values":[64,128,256,512,1280]}, 
                "n_inp": {"values":[64,128,256,512,1280]}, 
                "n_out": {"values":[64,128,256,512,1280]}, 
                "epochs": {"min": 10, "max": 300},
                "lr": {"max": 0.1, "min": 0.0001},
            },
        }

        sweep_id = wandb.sweep(sweep= sweep_configuration, project="my-first-sweep")
        wandb.agent(sweep_id, function=train, count=2)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrain', required=False, default=False, help="pretraining")
    parser.add_argument('-f', '--finetune', required=False, default=False, help="finetuning")
    parser.add_argument('-e', '--eval', required=False, default=False, help='evaluation')
    parser.add_argument('-t', '--hyperparameter_tuning', required=False, default=False, help='hyperparameter_tuning')
    args = parser.parse_args()

    main(args)


