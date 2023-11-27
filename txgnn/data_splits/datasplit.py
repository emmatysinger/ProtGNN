import numpy as np
import pandas as pd
import torch
from goatools.obo_parser import GODag
from goatools.godag.go_tasks import get_go2parents, get_go2children
import os
dirname = os.path.dirname(__file__)
    
class DataSplit:
      def __init__(self, kg_path=''): 
         self.kg, self.nodes, self.edges = self.load_kg(kg_path)
         self.edge_index = torch.LongTensor(self.edges.get(['x_index', 'y_index']).values.T)
         self.goid2parent, self.goid2children = self.load_go()
         
      def load_kg(self, pth=''):
         kg = pd.read_csv(pth+'kg.csv', low_memory=False)
         nodes = pd.read_csv(pth+'nodes.csv', low_memory=False)
         edges = pd.read_csv(pth+'edges.csv', low_memory=False)
         return kg, nodes, edges    
      
      def load_go(self, obo_path = os.path.join(dirname, 'go-basic.obo')):
          go_dag = GODag(obo_path)
          goid2parent = get_go2parents(go_dag, relationships=set())
          goid2children = get_go2children(go_dag, relationships=set())
          return goid2parent, goid2children
      
      def get_molfunc_nodes(self):
         self.molfunc_nodes = np.array(self.nodes[(self.nodes['node_type']=='molecular_function') & (self.nodes['node_source']=='GO')].node_index)
      
      def get_children(self, go_id, obo_file=os.path.join(dirname, 'go-basic.obo')):
         go_dag = GODag(obo_file)
         go_term = go_dag[go_id]

         children = go_term.get_all_children()
         children_data = [{'node_name': go_dag[child_id].name, 'go_id': child_id} for child_id in children]
         children_df = pd.DataFrame(children_data)

         return children_df

      def get_nodes_for_goid(self, code):
          children_df = self.get_children(code)
          if children_df.empty:
            children_df = pd.DataFrame(columns=['node_name', 'go_id'])
          merged_df = pd.merge(self.nodes, children_df, on='node_name', how='inner')
          return merged_df.get('node_index').values
      
      def get_edge_group(self, nodes, test_size = 0.05, add_prot_mf=True): 
         test_num_edges = round(self.edge_index.shape[1]*test_size)
         
         if add_prot_mf: 
            x = self.edges.query('x_index in @nodes or y_index in @nodes').query('relation=="molfunc_protein" or relation=="rev_molfunc_protein"')
            drug_dis_edges = x.get(['x_index','y_index']).values.T
            num_random_edges = test_num_edges - drug_dis_edges.shape[1]
         else: 
            num_random_edges = test_num_edges
            
         from torch_geometric.utils import k_hop_subgraph
         subgraph_nodes, filtered_edge_index, node_map, edge_mask = k_hop_subgraph(list(nodes), 2, self.edge_index) #one hop neighborhood
         sample_idx = np.random.choice(filtered_edge_index.shape[1], num_random_edges, replace=False)
         sample_edges = filtered_edge_index[:, sample_idx].numpy()
         
         if add_prot_mf:
            test_edges = np.concatenate([drug_dis_edges, sample_edges], axis=1)
         else: 
            test_edges = sample_edges
         
         test_edges = np.unique(test_edges, axis=1)
         return test_edges 
      
      def get_test_kg_for_disease(self, goid_code, test_size = 0.05, add_prot_mf=True): 
         molfunc_nodes = self.get_nodes_for_goid(goid_code)
         molfunc_edges = self.get_edge_group(molfunc_nodes, test_size = test_size, add_prot_mf=add_prot_mf)
         molfunc_edges = pd.DataFrame(molfunc_edges.T, columns=['x_index','y_index'])
         select_kg = pd.merge(self.kg, molfunc_edges, 'right').drop_duplicates()
         return select_kg
        
    
'''
Usage

ds = DataSplit(kg_path='../kg/kg_giant.csv')
test_kg = ds.get_test_kg_for_disease('GO:0016491')
'''

'''
Diseases selected for testing

    Code            Name                            # of nodes        

    GO:0005488      binding                         1794
    GO:0005215      transporter activity            950
    GO:0098772      molecular function regulator    226
    GO:0016491      oxidoreductase activity         2277
    GO:0016787      hydrolase activity              1344
    GO:0016740      transferase activity            2284
'''    