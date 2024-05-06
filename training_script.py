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

    node_types = ['anatomy', 
                  'biological_process', 
                  'cellular_component', 
                  'disease', 
                  'drug', 
                  'effect/phenotype', 
                  'exposure', 
                  'molecular_function', 
                  'pathway']

    node_type = 'pathway'

    print(f'[{get_timestamp()}] Loading data ...')
    TxData_inst = TxData(data_folder_path = data_path)
    TxData_inst.prepare_split(split = 'random', seed = 42, no_kg = False, remove_node_type = node_type)
    print(f'[{get_timestamp()}] Loaded data!')
    print(f'[{get_timestamp()}] Initializing model ...')


    if (args.pretrain or args.finetune or args.eval) and not args.hyperparameter_tuning:
        TxGNN_model = TxGNN(data = TxData_inst, 
                        weight_bias_track = True,
                        proj_name = 'MEng',
                        exp_name = f'ProtGNN Finetune No {node_type}',
                        device = 'cuda:0'
                        )
    
    if args.pretrain and not args.hyperparameter_tuning:
        n_inp_list = [128, 512, 1024]
        n_hid_list = [128, 512, 1024]
        n_out_list = [128, 512, 1024]

        TxGNN_model.model_initialize(n_hid = 512, #n_hid_list[int(args.n_hid)], #512
                                n_inp = 512, #n_inp_list[int(args.n_inp)], #1280
                                n_out = 1024, #n_out_list[int(args.n_out)], #512
                                proto = False, #made this False
                                proto_num = 3,
                                attention = False,
                                sim_measure = 'all_nodes_profile',
                                bert_measure = 'disease_name',
                                agg_measure = 'rarity',
                                num_walks = 200,
                                walk_mode = 'bit',
                                path_length = 2,
                                esm = False)
        print(f'[{get_timestamp()}] Initialized model!')

        print(f'[{get_timestamp()}] Starting to pretrain ...')
        TxGNN_model.pretrain(save_path = '/om/user/tysinger/models', 
                            name = f"protgnn_pretrain_no_{node_type.split('/')[0]}", 
                            n_epoch = 2, 
                            learning_rate = 1e-3,
                            batch_size = 512, 
                            train_print_per_n = 1000,
                            save_per_n = 4)
        print(f'[{get_timestamp()}] Pretrain done! ')

        TxGNN_model.save_model(f"/om/user/tysinger/models/protgnn_pretrain_no_{node_type.split('/')[0]}")
        #TxGNN_model.retrieve_embedding(path = '/om/user/tysinger/embeddings', save_name='pretrain_esm512_emb')

    if args.finetune and not args.hyperparameter_tuning:
        #wandb.init(project="MEng")
        if args.pretrain:
            TxGNN_model = TxGNN(data = TxData_inst, 
                    weight_bias_track = True,
                    proj_name = 'MEng',
                    exp_name = 'ProtGNN Finetuned',
                    device = 'cuda:0'
                    )
        TxGNN_model.load_pretrained(f"/om/user/tysinger/models/protgnn_pretrain_no_{node_type.split('/')[0]}", esm=False)

        print(f'[{get_timestamp()}] Starting to finetune ...')
        etype = 'BP'
        TxGNN_model.finetune(save_path = '/om/user/tysinger/models', 
                            name = f"protgnn_finetuned_no_{node_type.split('/')[0]}",
                            n_epoch = 150, 
                            learning_rate = 5e-4,
                            train_print_per_n = 5,
                            valid_per_n = 10,
                            save_per_n = 1000,
                            b=None,
                            edge_type = etype,
                            sweep_wandb=None)
        print(f'[{get_timestamp()}] Finetune done! ')
        TxGNN_model.save_model(f"/om/user/tysinger/models/protgnn_finetunedBP_no_{node_type.split('/')[0]}")
        TxGNN_model.retrieve_embedding(path = '/om/user/tysinger/embeddings', save_name=f"protgnn_finetuned_no_{node_type.split('/')[0]}")

    if args.eval:
        TxGNN_model.load_pretrained('/om/user/tysinger/models/esm_finetuned', esm=True)
        print(f'[{get_timestamp()}] Starting evaluation ...')

        TxEval_model = TxEval(model = TxGNN_model)
        #TxEval_model.load_eval_results(save_name = 'random_sigmoid_eval.pkl') 
        #TxEval_model.eval()


        #df_test = pd.read_csv('/om/user/tysinger/kg/random_42/test.csv')
        #filtered_df = df_test[df_test.x_type == 'molecular_function']
        #random_node_idxs = [int(i) for i in list(filtered_df.sample(n=200)['x_idx'])]


        result = TxEval_model.eval_molfunc_centric(molfunc_idxs = 'test_set', #'test_set'
                                            show_plot = 'esm_eval_finetune', 
                                            verbose = True, 
                                            save_result = True,
                                            save_name = 'esm_eval_finetune.pkl',
                                            return_raw = False)

    if args.hyperparameter_tuning:
        if args.pretrain:
            def train():
                wandb.init(project="Sweep")
                TxGNN_model = TxGNN(data = TxData_inst, 
                        weight_bias_track = False,
                        proj_name = 'MEng',
                        exp_name = 'Random with Flooding',
                        device = 'cuda:0'
                        )

                TxGNN_model.model_initialize(n_hid = 512, 
                                n_inp = 512, 
                                n_out = 512, 
                                proto = False, #made this False
                                proto_num = 3,
                                attention = False,
                                sim_measure = 'all_nodes_profile',
                                bert_measure = 'disease_name',
                                agg_measure = 'rarity',
                                num_walks = 200,
                                walk_mode = 'bit',
                                path_length = 2)

                TxGNN_model.pretrain(save_path = '/om/user/tysinger/models', 
                            name = 'pretrain_esm', 
                            n_epoch = wandb.config.epochs, 
                            learning_rate = wandb.config.lr,
                            batch_size = wandb.config.batch_size, 
                            train_print_per_n = 500,
                            save_per_n = 1,
                            sweep_wandb = wandb)
                
                TxGNN_model.finetune(n_epoch = 150, 
                        learning_rate = 5e-4,
                        train_print_per_n = 500,
                        valid_per_n = 5,
                        save_path = '/om/user/tysinger/models', 
                        name = 'sweep',
                        sweep_wandb = wandb,
                        b=0.2)
            
            sweep_configuration = {
                "method": "grid",
                "name": "sweep_pretrain_val_loss",
                "metric": {"goal": "minimize", "name": "validation_loss"},
                "parameters": {
                    "batch_size": {"values": [256, 512, 1024, 2048]},
                    "epochs": {"values": [1,2,3]},
                    "lr": {"values":[1e-3, 1e-4, 5e-4]},
                },
            }

            #sweep_id = wandb.sweep(sweep= sweep_configuration, project="Sweep")
            sweep_id = 'jk3krlqe'
            wandb.agent(sweep_id, function=train, count=3, project="Sweep")

        if args.finetune:
            def train():
                wandb.init(project="Sweep_Test")
                TxGNN_model = TxGNN(data = TxData_inst, 
                        weight_bias_track = False,
                        proj_name = 'MEng',
                        exp_name = 'Random with Flooding',
                        device = 'cuda:0'
                        )
                
                TxGNN_model.load_pretrained(f'/om/user/tysinger/models/pretrained_for_finetune_hyper/pretrain_{wandb.config.n_inp}_{wandb.config.n_hid}_{wandb.config.n_out}')

                etype = 'MF'
                TxGNN_model.finetune(n_epoch = wandb.config.epochs, 
                        learning_rate = wandb.config.lr,
                        train_print_per_n = 500,
                        valid_per_n = 5,
                        save_path = '/om/user/tysinger/models', 
                        name = 'finetune',
                        sweep_wandb = wandb,
                        b=None, 
                        edge_type = etype)
            
            sweep_configuration = {
                "method": "bayes",
                "name": "Finetune",
                "metric": {"goal": "minimize", "name": "validation_loss"},
                "parameters": {
                    "n_inp": {"values":[128,512,1024]}, #64, 256, 1280
                    "n_hid": {"values":[128,512,1024]}, 
                    "n_out": {"values":[128,512,1024]}, 
                    "epochs": {"min": 10, "max": 300},
                    "lr": {"max": 0.01, "min": 0.00005},
                },
            }
            #sweep_id = wandb.sweep(sweep= sweep_configuration, project="Sweep")
            sweep_id = 'zwcjfsg7'
            wandb.agent(sweep_id, function=train, count=5, project = 'Sweep')
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrain', required=False, default=False, help="pretraining")
    parser.add_argument('-f', '--finetune', required=False, default=False, help="finetuning")
    parser.add_argument('-e', '--eval', required=False, default=False, help='evaluation')
    parser.add_argument('-t', '--hyperparameter_tuning', required=False, default=False, help='hyperparameter_tuning')
    parser.add_argument('--n_inp', required=False, default=None, help='n_inp dimensions')
    parser.add_argument('--n_hid', required=False, default=None, help='n_hid dimensions')
    parser.add_argument('--n_out', required=False, default=None, help='n_out dimensions')
    args = parser.parse_args()

    main(args)


