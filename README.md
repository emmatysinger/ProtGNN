# ProtGNN: Zero-shot prediction of therapeutic use with geometric deep learning and human centered design

This repository hosts the official implementation of ProtGNN, a model for learning protein representations that encode biomedical domain information about proteins using a biomedical knowledge graph. 

### **Training**

`training_script.py` is the main script to pretrain and finetune ProtGNN. Be sure to changes file paths and wandb login parameters within this file before running. Arguments:

- `-p`/`--pretrain`: True or False (False by default)
- `-f`/`--finetune`: True or False (False by default)
- `-e`/`--eval`: True or False (False by default)
- `-h`/`--hyperparameter_tuning`: True or False (False by default)
- `--n_inp`: number of input dimensions (None by default)
- `--n_hid`: number of hidden dimensions (None by default)
- `--n_out`: number of output dimensions (None by default)

The script can be run from the command line or in a bash script for pretraining like this:

```
python training_script.py -p
```

or for finetuning like this:

```
python training_script.py -f
```

### **Embedding Space Visualization**

Function to visualize the embedding spaces are found in `visualize_utils.py`. 
Label options include 'biological_process' and 'molecular_function'.

An example of visualizing the embedding space by 'biological process' labels: 

```
from txgnn import TxData
from visualize_utils import GO, Embeddings, visualize_pipeline

embed_path = '/PATH/TO/embeddings.pkl'

TxData_inst = TxData(data_folder_path = '/PATH/TO/PrimeKG/')
TxData_inst.prepare_split(split = 'random', seed = 42, no_kg = False)

visualize_pipeline(embed_path=embed_path, node_type = 'biological_process', TxData_inst=TxData_inst, kmeans=True)
```



