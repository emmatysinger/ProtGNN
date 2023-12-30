from txgnn import TxData, TxGNN

TxData_inst = TxData(data_folder_path = '/om/user/tysinger/kg/')
TxData_inst.prepare_split(split = 'random', seed = 42, no_kg = False)
TxGNN_model = TxGNN(data = TxData_inst, 
            weight_bias_track = False,
            proj_name = 'MEng',
            exp_name = 'Sweep',
            device = 'cuda:0'
            )
TxGNN_model.load_pretrained('/om/user/tysinger/models/random_finetuned', esm=False)
h = TxGNN_model.retrieve_embedding(path = '/om/user/tysinger/models')
print(h['gene/protein'])