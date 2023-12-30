from txgnn import TxData, TxGNN, TxEval
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.decomposition import SparsePCA, TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np
from scipy import sparse
import torch
#
TxData_inst = TxData(data_folder_path = '/om/user/tysinger/kg/')
TxData_inst.prepare_split(split = 'random', seed = 42, no_kg = False)

all_protein_profiles = []
for idx, prot_idx in enumerate(TxData_inst.G.nodes('gene/protein').tolist()):
    nodes = TxData_inst.G.successors(prot_idx, etype='molfunc_protein')
    num_nodes = len(TxData_inst.G.nodes('molecular_function'))
    node_profile = torch.zeros((num_nodes,))
    node_profile[nodes] = 1.
    all_protein_profiles.append(node_profile.tolist())

with open(os.path.join('/om/user/tysinger/embeddings', 'protein_function_labels.pkl'), 'wb') as f:
    pickle.dump(all_protein_profiles, f)


sparse_profiles = sparse.csr_matrix(np.array(all_protein_profiles))
truncatedsvd = TruncatedSVD(n_components = 50, algorithm='randomized')
svd_profiles = truncatedsvd.fit_transform(sparse_profiles)

kmeans = KMeans(n_clusters = 45, random_state = 42)
clusters = kmeans.fit_predict(all_protein_profiles)

print(max(clusters))

with open(os.path.join('/om/user/tysinger/embeddings', 'protein_function_clusters.pkl'), 'wb') as f:
    pickle.dump(clusters, f)



